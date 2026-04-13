from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import yaml
from pyspark.sql import DataFrame, functions as F


@dataclass(frozen=True)
class DqResult:
    passed: bool
    metrics: dict[str, Any]
    failed_frames: dict[str, DataFrame]


def _not_null_failures(df: DataFrame, columns: list[str]) -> DataFrame:
    expr = None
    for col_name in columns:
        current = F.col(col_name).isNull()
        expr = current if expr is None else (expr | current)

    if expr is None:
        raise ValueError("not_null check requires at least one column")

    return df.filter(expr)


def _unique_failures(df: DataFrame, columns: list[str]) -> tuple[DataFrame, int]:
    dup_keys = df.groupBy(*columns).count().filter(F.col("count") > 1).drop("count")
    duplicate_key_count = dup_keys.count()
    failed_rows = df.join(dup_keys, on=columns, how="inner")
    return failed_rows, duplicate_key_count


def _in_set_failures(df: DataFrame, column: str, allowed: list[str]) -> DataFrame:
    return df.filter(F.col(column).isNull() | (~F.col(column).isin(allowed)))


def run_dq(df: DataFrame, expectations_path: str) -> DqResult:
    with open(expectations_path, "r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    metrics: dict[str, Any] = {
        "dataset": spec.get("dataset"),
        "layer": spec.get("layer"),
        "version": spec.get("version"),
        "row_count": df.count(),
        "checks": [],
    }
    failed_frames: dict[str, DataFrame] = {}
    passed = True

    for chk in spec["checks"]:
        ctype = chk["type"]
        name = chk["name"]
        severity = chk.get("severity", "warning")
        detail: dict[str, Any] = {"name": name, "type": ctype, "severity": severity}

        if ctype == "not_null":
            failed_df = _not_null_failures(df, chk["columns"])
            bad_rows = failed_df.count()
            detail["bad_rows"] = bad_rows
            ok = bad_rows == 0

        elif ctype == "unique":
            failed_df, duplicate_key_count = _unique_failures(df, chk["columns"])
            bad_rows = failed_df.count()
            detail["bad_rows"] = bad_rows
            detail["duplicate_keys"] = duplicate_key_count
            ok = duplicate_key_count == 0

        elif ctype == "in_set":
            failed_df = _in_set_failures(df, chk["column"], chk["allowed"])
            bad_rows = failed_df.count()
            detail["bad_rows"] = bad_rows
            ok = bad_rows == 0

        else:
            raise ValueError(f"Unknown check type: {ctype}")

        detail["passed"] = ok
        metrics["checks"].append(detail)

        if not ok:
            failed_frames[name] = failed_df
            if severity == "critical":
                passed = False

    return DqResult(passed=passed, metrics=metrics, failed_frames=failed_frames)


def write_dq_report(result: DqResult, report_path: str) -> None:
    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True), encoding="utf-8")


def write_failed_rows(
    result: DqResult,
    quarantine_base_path: str,
    sample_limit: int = 100,
) -> list[str]:
    base = Path(quarantine_base_path)
    base.mkdir(parents=True, exist_ok=True)

    written_paths: list[str] = []
    for check_name, failed_df in result.failed_frames.items():
        out_path = str(base / check_name)
        (
            failed_df.limit(sample_limit)
            .write.mode("overwrite")
            .parquet(out_path)
        )
        written_paths.append(out_path)

    return written_paths