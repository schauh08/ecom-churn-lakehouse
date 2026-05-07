from __future__ import annotations

from pyspark.sql import functions as F

from src.transformations.orders_bronze_to_silver import normalize_and_dedupe_orders


def test_normalize_and_dedupe_orders_keeps_latest_valid_record(spark) -> None:
    rows = [
        {
            "order_id": "ORD-1",
            "customer_id": "CUST-1",
            "order_purchase_timestamp": "2025-01-01 10:00:00",
            "order_status": "delivered",
            "run_id": "run-1",
            "ingest_ts": "2025-01-01 12:00:00",
            "source_file": "a.parquet",
            "source_fingerprint": "fp1",
            "schema_hash": "sh1",
        },
        {
            "order_id": "ORD-1",
            "customer_id": "CUST-1",
            "order_purchase_timestamp": "2025-01-02 10:00:00",
            "order_status": "shipment_pending",
            "run_id": "run-2",
            "ingest_ts": "2025-01-02 12:00:00",
            "source_file": "b.parquet",
            "source_fingerprint": "fp2",
            "schema_hash": "sh2",
        },
        {
            "order_id": "ORD-2",
            "customer_id": None,
            "order_purchase_timestamp": "2025-01-03 10:00:00",
            "order_status": "delivered",
            "run_id": "run-3",
            "ingest_ts": "2025-01-03 12:00:00",
            "source_file": "c.parquet",
            "source_fingerprint": "fp3",
            "schema_hash": "sh3",
        },
    ]

    bronze = (
        spark.createDataFrame(rows)
        .withColumn("ingest_ts", F.to_timestamp("ingest_ts"))
    )

    deduped, invalid_rows, duplicate_rejects = normalize_and_dedupe_orders(
        bronze,
        allowed_statuses=[
            "approved",
            "canceled",
            "created",
            "delivered",
            "invoiced",
            "processing",
            "shipped",
            "unavailable",
        ],
    )

    assert deduped.count() == 1
    assert invalid_rows.count() == 1
    assert duplicate_rejects.count() == 1

    row = deduped.collect()[0]
    assert row["order_id"] == "ord-1"
    assert row["customer_id"] == "cust-1"
    assert row["order_status"] == "processing"
    assert str(row["_bronze_run_id"]) == "run-2"