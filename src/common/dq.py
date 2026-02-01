from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
import yaml
from pyspark.sql import DataFrame, functions as F

@dataclass(frozen=True)
class DqResult:
    passed: bool
    metrics: Dict[str, Any]

def run_dq(df: DataFrame, expectations_path: str) -> DqResult:
    spec = yaml.safe_load(open(expectations_path, "r", encoding="utf-8"))
    metrics: Dict[str, Any] = {"checks": []}
    passed = True

    for chk in spec["checks"]:
        ctype = chk["type"]
        name = chk["name"]
        severity = chk.get("severity", "warning")
        ok = True
        detail: Dict[str, Any] = {"name": name, "type": ctype, "severity": severity}

        if ctype == "not_null":
            cols = chk["columns"]
            expr = None
            for col in cols:
                expr = F.col(col).isNotNull() if expr is None else (expr & F.col(col).isNotNull())
            bad = df.filter(~expr).count()
            ok = bad == 0
            detail["bad_rows"] = bad

        elif ctype == "unique":
            cols = chk["columns"]
            dup = (
                df.groupBy([F.col(c) for c in cols])
                .count()
                .filter(F.col("count") > 1)
                .count()
            )
            ok = dup == 0
            detail["duplicate_keys"] = dup

        elif ctype == "in_set":
            col = chk["column"]
            allowed = chk["allowed"]
            bad = df.filter(~F.col(col).isin(allowed)).count()
            ok = bad == 0
            detail["bad_rows"] = bad

        else:
            raise ValueError(f"Unknown check type: {ctype}")

        detail["passed"] = ok
        metrics["checks"].append(detail)

        if severity == "critical" and not ok:
            passed = False

    return DqResult(passed=passed, metrics=metrics)
