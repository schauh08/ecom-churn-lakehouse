from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.common.spark import get_spark


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str], timeout_s: int = 1200) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout_s,
    )
    if proc.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(cmd)}\n\n{proc.stdout}")


def write_sample_orders_parquet(path: Path) -> None:
    rows = [
        {
            "order_id": "A1",
            "customer_id": "CUST_0001",
            "order_status": "delivered",
            "order_purchase_timestamp": "2025-01-10 10:00:00",
            "order_approved_at": None,
            "order_delivered_carrier_date": None,
            "order_delivered_customer_date": None,
            "order_estimated_delivery_date": None,
        },
        {
            "order_id": "B1",
            "customer_id": "CUST_0002",
            "order_status": "delivered",
            "order_purchase_timestamp": "2025-01-20 12:00:00",
            "order_approved_at": None,
            "order_delivered_carrier_date": None,
            "order_delivered_customer_date": None,
            "order_estimated_delivery_date": None,
        },
        {
            "order_id": "C1",
            "customer_id": "CUST_0003",
            "order_status": "delivered",
            "order_purchase_timestamp": "2025-02-15 09:00:00",
            "order_approved_at": None,
            "order_delivered_carrier_date": None,
            "order_delivered_customer_date": None,
            "order_estimated_delivery_date": None,
        },
        {
            "order_id": "A2",
            "customer_id": "CUST_0001",
            "order_status": "delivered",
            "order_purchase_timestamp": "2025-03-10 11:00:00",
            "order_approved_at": None,
            "order_delivered_carrier_date": None,
            "order_delivered_customer_date": None,
            "order_estimated_delivery_date": None,
        },
        {
            "order_id": "C2",
            "customer_id": "CUST_0003",
            "order_status": "delivered",
            "order_purchase_timestamp": "2025-04-10 15:00:00",
            "order_approved_at": None,
            "order_delivered_carrier_date": None,
            "order_delivered_customer_date": None,
            "order_estimated_delivery_date": None,
        },
        {
            "order_id": "A3",
            "customer_id": "CUST_0001",
            "order_status": "delivered",
            "order_purchase_timestamp": "2025-05-10 08:30:00",
            "order_approved_at": None,
            "order_delivered_carrier_date": None,
            "order_delivered_customer_date": None,
            "order_estimated_delivery_date": None,
        },
        {
            "order_id": "Z1",
            "customer_id": "CUST_9999",
            "order_status": "delivered",
            "order_purchase_timestamp": "2025-06-15 00:00:00",
            "order_approved_at": None,
            "order_delivered_carrier_date": None,
            "order_delivered_customer_date": None,
            "order_estimated_delivery_date": None,
        },
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


@pytest.mark.e2e
def test_slice_e2e(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    python = sys.executable
    run_id = str(uuid.uuid4())

    sandbox = tmp_path / "sandbox"
    lakehouse_root = sandbox / "lakehouse"
    artifacts_root = sandbox / "artifacts"

    bronze_orders = lakehouse_root / "bronze" / "orders"
    silver_orders = lakehouse_root / "silver" / "orders"
    gold_features = lakehouse_root / "gold" / "customer_features_daily"
    gold_labels = lakehouse_root / "gold" / "customer_labels_daily"
    training_snapshot = lakehouse_root / "training" / "customer_training_snapshot"
    latest_features = artifacts_root / "serving" / "latest_features"

    model_path = artifacts_root / "models" / "ecomm_churn_baseline.pkl"
    model_meta = artifacts_root / "models" / "model_meta.json"
    eval_summary = artifacts_root / "models" / "evaluation_summary.md"
    approved_model = artifacts_root / "models" / "approved_model_version.json"
    feature_schema = artifacts_root / "models" / "feature_schema_info.json"

    sample_parquet = sandbox / "orders.parquet"
    write_sample_orders_parquet(sample_parquet)

    env = dict(os.environ)
    env.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    env["PYSPARK_SUBMIT_ARGS"] = (
        f"--conf spark.ui.enabled=false "
        f"--conf spark.sql.shuffle.partitions=4 "
        f"--conf spark.sql.warehouse.dir={sandbox / 'spark-warehouse'} "
        "pyspark-shell"
    )

    run_cmd(
        [
            python,
            "-m",
            "src.ingestion.orders_to_bronze",
            "--input",
            str(sample_parquet),
            "--bronze_path",
            str(bronze_orders),
            "--run_id",
            run_id,
        ],
        cwd=repo_root,
        env=env,
    )

    run_cmd(
        [
            python,
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
        ],
        cwd=repo_root,
        env=env,
    )

    as_of_dates = ["2025-01-31", "2025-02-28", "2025-03-31"]

    for as_of_date in as_of_dates:
        run_cmd(
            [
                python,
                "-m",
                "src.features.customer_features_daily",
                "--silver_path",
                str(silver_orders),
                "--gold_path",
                str(gold_features),
                "--contract",
                str(repo_root / "data" / "contracts" / "gold" / "customer_features_daily.v1.json"),
                "--as_of_date",
                as_of_date,
                "--run_id",
                f"{run_id}-gold-{as_of_date}",
            ],
            cwd=repo_root,
            env=env,
        )

        run_cmd(
            [
                python,
                "-m",
                "src.training.labels",
                "--silver_path",
                str(silver_orders),
                "--labels_path",
                str(gold_labels),
                "--as_of_date",
                as_of_date,
                "--run_id",
                f"{run_id}-labels-{as_of_date}",
                "--metadata_path",
                str(artifacts_root / "labels" / f"{as_of_date}.json"),
            ],
            cwd=repo_root,
            env=env,
        )

    run_cmd(
        [
            python,
            "-m",
            "src.training.build_training_snapshot",
            "--gold_path",
            str(gold_features),
            "--labels_path",
            str(gold_labels),
            "--training_snapshot_path",
            str(training_snapshot),
            "--run_id",
            f"{run_id}-snapshot",
            "--metadata_path",
            str(artifacts_root / "training" / "snapshot.json"),
        ],
        cwd=repo_root,
        env=env,
    )

    run_cmd(
        [
            python,
            "-m",
            "src.training.train_stub",
            "--training_snapshot_path",
            str(training_snapshot),
            "--feature_contract",
            str(repo_root / "data" / "contracts" / "gold" / "customer_features_daily.v1.json"),
            "--out_model_path",
            str(model_path),
            "--out_model_meta",
            str(model_meta),
            "--evaluation_summary_path",
            str(eval_summary),
            "--approved_model_version_path",
            str(approved_model),
            "--feature_schema_info_path",
            str(feature_schema),
            "--model_name",
            "ecomm-churn",
            "--validation_fraction",
            "0.34",
            "--mlflow_tracking_uri",
            f"file:{artifacts_root / 'mlruns'}",
            "--mlflow_experiment",
            "ecomm-churn-e2e",
        ],
        cwd=repo_root,
        env=env,
    )

    run_cmd(
        [
            python,
            "-m",
            "src.serving_features.build_latest_features",
            "--gold_path",
            str(gold_features),
            "--latest_features_path",
            str(latest_features),
            "--run_id",
            f"{run_id}-serving",
            "--manifest_path",
            str(artifacts_root / "serving" / "manifest.json"),
        ],
        cwd=repo_root,
        env=env,
    )

    assert model_path.exists()
    assert model_meta.exists()
    assert approved_model.exists()
    assert latest_features.exists()

    meta = json.loads(model_meta.read_text(encoding="utf-8"))
    assert meta["model_version"]
    assert meta["feature_version"]

    spark = get_spark("slice_e2e_assertions")
    try:
        gold_df = spark.read.format("delta").load(str(gold_features))
        assert {"customer_id", "as_of_date", "_feature_version"}.issubset(set(gold_df.columns))
    finally:
        spark.stop()

    monkeypatch.setenv("API_KEY", "test-api-key")
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_META_PATH", str(model_meta))
    monkeypatch.setenv("APPROVED_MODEL_VERSION_PATH", str(approved_model))
    monkeypatch.setenv("LATEST_FEATURES_PATH", str(latest_features))

    from services.api.app.routers.predict import get_feature_store, get_model_store

    get_feature_store.cache_clear()
    get_model_store.cache_clear()

    from services.api.app.main import app

    client = TestClient(app)
    response = client.post(
        "/v1/churn/predict",
        headers={"X-API-Key": "test-api-key"},
        json={"customer_id": "cust_0001"},
    )

    assert response.status_code == 200, response.text
    body = response.json()

    assert body["customer_id"] == "cust_0001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_label"] in [0, 1]
    assert body["model_version"]
    assert body["feature_version"]
    assert body["request_id"]