from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, Sequence, Tuple

import pytest


def run_cmd(cmd: Sequence[str], *, cwd: Path, env: Dict[str, str], timeout_s: int = 900) -> None:
    """Run a command and fail fast with full combined logs."""
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(cmd)}\n\n{proc.stdout}")


def assert_dir_nonempty(p: Path) -> None:
    assert p.exists(), f"Missing path: {p}"
    assert p.is_dir(), f"Expected a directory: {p}"
    assert any(p.rglob("*")), f"Directory is empty: {p}"


def read_table_columns(path: Path) -> Tuple[list[str], str]:
    """
    Reads schema columns from Gold table directory.
    Supports Delta (if _delta_log exists) or Parquet directory.
    Returns (columns, format_name).
    """
    # Prefer Spark because Delta might be used.
    from pyspark.sql import SparkSession  # type: ignore

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("slice_e2e_test")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    try:
        if (path / "_delta_log").exists():
            df = spark.read.format("delta").load(str(path))
            fmt = "delta"
        else:
            df = spark.read.parquet(str(path))
            fmt = "parquet"
        return df.columns, fmt
    finally:
        spark.stop()


def read_distinct_feature_version(path: Path) -> str | None:
    """Optional stronger check: feature_version stamped in Gold matches meta.json."""
    from pyspark.sql import SparkSession  # type: ignore

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("slice_e2e_test_feature_version")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    try:
        if (path / "_delta_log").exists():
            df = spark.read.format("delta").load(str(path))
        else:
            df = spark.read.parquet(str(path))

        if "_feature_version" not in df.columns:
            return None

        row = df.select("_feature_version").where(df["_feature_version"].isNotNull()).limit(1).collect()
        return row[0]["_feature_version"] if row else None
    finally:
        spark.stop()


@pytest.mark.e2e
def test_slice_e2e(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]

    run_id = str(uuid.uuid4())

    # --- sandbox paths (no writes to repo) ---
    sandbox = tmp_path / "sandbox"
    lakehouse_root = sandbox / "lakehouse"
    artifacts_root = sandbox / "artifacts"

    bronze_orders = lakehouse_root / "bronze" / "orders"
    silver_orders = lakehouse_root / "silver" / "orders"
    gold_features = lakehouse_root / "gold" / "customer_features_daily"
    model_meta = artifacts_root / "model_meta.json"

    for d in (bronze_orders, silver_orders, gold_features, artifacts_root):
        d.mkdir(parents=True, exist_ok=True)

    # --- sample input: copy your existing parquet into sandbox ---
    sample_src = repo_root / "data" / "sample" / "orders.parquet"
    assert sample_src.exists(), f"Sample input missing: {sample_src}"
    sample_dst = sandbox / "orders.parquet"
    shutil.copyfile(sample_src, sample_dst)

    # Keep env controlled. (You can add SPARK_* here if needed.)
    env = dict(os.environ)

    # --- run the SAME commands as your Makefile ---
    commands = [
        (
            "python",
            "-m",
            "src.ingestion.orders_to_bronze",
            "--input",
            str(sample_dst),
            "--bronze_path",
            str(bronze_orders),
            "--run_id",
            run_id,
        ),
        (
            "python",
            "-m",
            "src.transformations.orders_bronze_to_silver",
            "--bronze_path",
            str(bronze_orders),
            "--silver_path",
            str(silver_orders),
            "--contract",
            str(repo_root / "data" / "contracts" / "silver" / "orders.v1.json"),
            "--expectations",
            str(repo_root / "data" / "expectations" / "silver" / "orders.yml"),
            "--run_id",
            run_id,
        ),
        (
            "python",
            "-m",
            "src.features.customer_features_daily",
            "--silver_path",
            str(silver_orders),
            "--gold_path",
            str(gold_features),
            "--contract",
            str(repo_root / "data" / "contracts" / "gold" / "customer_features_daily.v1.json"),
            "--as_of_date",
            "2026-01-31",
            "--run_id",
            run_id,
        ),
        (
            "python",
            "-m",
            "src.training.train_stub",
            "--feature_contract",
            str(repo_root / "data" / "contracts" / "gold" / "customer_features_daily.v1.json"),
            "--out_model_meta",
            str(model_meta),
        ),
    ]

    for cmd in commands:
        run_cmd(cmd, cwd=repo_root, env=env, timeout_s=900)

    # --- assertions: lock the spine contract ---
    assert_dir_nonempty(bronze_orders)
    assert_dir_nonempty(silver_orders)
    assert_dir_nonempty(gold_features)

    cols, fmt = read_table_columns(gold_features)
    required = {"customer_id", "as_of_date", "_feature_version"}
    missing = required - set(cols)
    assert not missing, f"Gold ({fmt}) missing columns {missing}. Found columns={cols}"

    assert model_meta.exists(), f"Missing model meta: {model_meta}"
    meta = json.loads(model_meta.read_text())

    assert meta.get("model_version"), f"model_version missing/empty in {meta}"
    assert meta.get("feature_version"), f"feature_version missing/empty in {meta}"

    # Optional but strong: ensure Gold stamped version matches meta
    fv_gold = read_distinct_feature_version(gold_features)
    if fv_gold is not None:
        assert fv_gold == meta["feature_version"], (
            f"feature_version mismatch: gold={fv_gold} vs meta={meta['feature_version']}"
        )
