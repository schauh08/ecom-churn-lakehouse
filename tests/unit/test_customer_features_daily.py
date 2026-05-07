from __future__ import annotations

from pyspark.sql import functions as F

from src.features.customer_features_daily import build_feature_snapshot


def test_build_feature_snapshot_computes_expected_features(spark) -> None:
    rows = [
        {
            "customer_id": "cust_1",
            "order_id": "o1",
            "order_purchase_ts": "2025-01-10 10:00:00",
        },
        {
            "customer_id": "cust_1",
            "order_id": "o2",
            "order_purchase_ts": "2025-03-10 10:00:00",
        },
        {
            "customer_id": "cust_2",
            "order_id": "o3",
            "order_purchase_ts": "2025-03-20 10:00:00",
        },
    ]

    silver_orders = (
        spark.createDataFrame(rows)
        .withColumn("order_purchase_ts", F.to_timestamp("order_purchase_ts"))
    )

    features = build_feature_snapshot(
        silver_orders,
        as_of_date="2025-03-31",
        snapshot_id="snap123",
        feature_version="feat123",
        run_id="gold-run-1",
    )

    by_customer = {row["customer_id"]: row.asDict() for row in features.collect()}

    assert set(by_customer) == {"cust_1", "cust_2"}

    c1 = by_customer["cust_1"]
    assert c1["recency_days"] == 21
    assert c1["orders_30d"] == 1
    assert c1["orders_90d"] == 2
    assert c1["lifetime_orders"] == 2
    assert c1["customer_tenure_days"] == 80
    assert round(c1["avg_days_between_orders"], 2) == 59.00
    assert c1["_snapshot_id"] == "snap123"
    assert c1["_feature_version"] == "feat123"

    c2 = by_customer["cust_2"]
    assert c2["recency_days"] == 11
    assert c2["orders_30d"] == 1
    assert c2["orders_90d"] == 1
    assert c2["lifetime_orders"] == 1
    assert c2["customer_tenure_days"] == 11
    assert c2["avg_days_between_orders"] == 0.0