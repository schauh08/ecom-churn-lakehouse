from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.common.spark import get_spark
from src.common.versioning import hash_contract_json


def _build_snapshot_id(as_of_date: str, feature_version: str) -> str:
    payload = f"{as_of_date}|{feature_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _delta_exists(spark, path: str) -> bool:
    try:
        return DeltaTable.isDeltaTable(spark, path)
    except Exception:
        return False


def _assert_gold_quality(features: DataFrame) -> dict[str, int]:
    required_null_expr = (
        F.col("customer_id").isNull()
        | F.col("as_of_date").isNull()
        | F.col("recency_days").isNull()
        | F.col("orders_30d").isNull()
        | F.col("orders_90d").isNull()
        | F.col("lifetime_orders").isNull()
        | F.col("customer_tenure_days").isNull()
        | F.col("avg_days_between_orders").isNull()
        | F.col("_snapshot_id").isNull()
        | F.col("_feature_version").isNull()
        | F.col("_gold_run_id").isNull()
        | F.col("_gold_ts").isNull()
    )

    null_rows = features.filter(required_null_expr).count()

    duplicate_keys = (
        features.groupBy("customer_id", "as_of_date")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    invalid_ranges = features.filter(
        (F.col("recency_days") < 0)
        | (F.col("orders_30d") < 0)
        | (F.col("orders_90d") < 0)
        | (F.col("lifetime_orders") < 1)
        | (F.col("customer_tenure_days") < 0)
        | (F.col("avg_days_between_orders") < 0)
    ).count()

    invalid_ordering = features.filter(
        (F.col("orders_30d") > F.col("orders_90d"))
        | (F.col("orders_90d") > F.col("lifetime_orders"))
        | (F.col("recency_days") > F.col("customer_tenure_days"))
    ).count()

    metrics = {
        "null_rows": null_rows,
        "duplicate_keys": duplicate_keys,
        "invalid_ranges": invalid_ranges,
        "invalid_ordering": invalid_ordering,
    }

    if any(value > 0 for value in metrics.values()):
        raise RuntimeError(f"Gold feature quality checks failed: {metrics}")

    return metrics


def _write_snapshot_metadata(
    metadata_path: str | None,
    *,
    snapshot_id: str,
    as_of_date: str,
    feature_version: str,
    run_id: str,
    row_count: int,
    quality_metrics: dict[str, int],
) -> None:
    if not metadata_path:
        return

    payload = {
        "snapshot_id": snapshot_id,
        "as_of_date": as_of_date,
        "feature_version": feature_version,
        "gold_run_id": run_id,
        "row_count": row_count,
        "quality_metrics": quality_metrics,
    }

    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_feature_snapshot(
    silver_orders: DataFrame,
    *,
    as_of_date: str,
    snapshot_id: str,
    feature_version: str,
    run_id: str,
) -> DataFrame:
    as_of_date_col = F.to_date(F.lit(as_of_date))

    orders = (
        silver_orders.select(
            "customer_id",
            "order_id",
            "order_purchase_ts",
        )
        .withColumn("order_date", F.to_date("order_purchase_ts"))
    )

    historical = orders.filter(F.col("order_date") <= as_of_date_col)
    customers = historical.select("customer_id").distinct()

    order_stats = historical.groupBy("customer_id").agg(
        F.max("order_date").alias("last_order_date"),
        F.min("order_date").alias("first_order_date"),
        F.countDistinct("order_id").cast("long").alias("lifetime_orders"),
    )

    orders_30d = (
        historical.filter(F.col("order_date") >= F.date_sub(as_of_date_col, 29))
        .groupBy("customer_id")
        .agg(F.countDistinct("order_id").cast("long").alias("orders_30d"))
    )

    orders_90d = (
        historical.filter(F.col("order_date") >= F.date_sub(as_of_date_col, 89))
        .groupBy("customer_id")
        .agg(F.countDistinct("order_id").cast("long").alias("orders_90d"))
    )

    gap_window = Window.partitionBy("customer_id").orderBy(
        F.col("order_purchase_ts").asc(),
        F.col("order_id").asc(),
    )

    gaps = (
        historical.select("customer_id", "order_id", "order_purchase_ts")
        .withColumn("prev_order_ts", F.lag("order_purchase_ts").over(gap_window))
        .withColumn(
            "gap_days",
            F.when(
                F.col("prev_order_ts").isNull(),
                F.lit(None),
            ).otherwise(
                F.datediff(
                    F.to_date(F.col("order_purchase_ts")),
                    F.to_date(F.col("prev_order_ts")),
                )
            ),
        )
    )

    avg_gaps = gaps.groupBy("customer_id").agg(
        F.avg("gap_days").cast("double").alias("avg_days_between_orders")
    )

    return (
        customers.join(order_stats, on="customer_id", how="inner")
        .join(orders_30d, on="customer_id", how="left")
        .join(orders_90d, on="customer_id", how="left")
        .join(avg_gaps, on="customer_id", how="left")
        .fillna(
            {
                "orders_30d": 0,
                "orders_90d": 0,
                "avg_days_between_orders": 0.0,
            }
        )
        .withColumn("as_of_date", as_of_date_col)
        .withColumn("recency_days", F.datediff(as_of_date_col, F.col("last_order_date")))
        .withColumn(
            "customer_tenure_days",
            F.datediff(as_of_date_col, F.col("first_order_date")),
        )
        .withColumn("_snapshot_id", F.lit(snapshot_id))
        .withColumn("_feature_version", F.lit(feature_version))
        .withColumn("_gold_run_id", F.lit(run_id))
        .withColumn("_gold_ts", F.current_timestamp())
        .select(
            "customer_id",
            "as_of_date",
            "recency_days",
            "orders_30d",
            "orders_90d",
            "lifetime_orders",
            "customer_tenure_days",
            "avg_days_between_orders",
            "_snapshot_id",
            "_feature_version",
            "_gold_run_id",
            "_gold_ts",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver_path", required=True)
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--as_of_date", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--snapshot_metadata_path",
        default=None,
        help="Optional JSON path for snapshot metadata artifact.",
    )
    args = parser.parse_args()

    spark = get_spark("customer_features_daily")

    feature_version = hash_contract_json(args.contract)
    snapshot_id = _build_snapshot_id(args.as_of_date, feature_version)

    silver_orders = spark.read.format("delta").load(args.silver_path)

    features = build_feature_snapshot(
        silver_orders,
        as_of_date=args.as_of_date,
        snapshot_id=snapshot_id,
        feature_version=feature_version,
        run_id=args.run_id,
    )

    quality_metrics = _assert_gold_quality(features)
    row_count = features.count()

    _write_snapshot_metadata(
        args.snapshot_metadata_path,
        snapshot_id=snapshot_id,
        as_of_date=args.as_of_date,
        feature_version=feature_version,
        run_id=args.run_id,
        row_count=row_count,
        quality_metrics=quality_metrics,
    )

    if _delta_exists(spark, args.gold_path):
        target = DeltaTable.forPath(spark, args.gold_path)
        (
            target.alias("t")
            .merge(
                features.alias("s"),
                "t.customer_id = s.customer_id AND t.as_of_date = s.as_of_date",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        features.write.format("delta").mode("overwrite").save(args.gold_path)


if __name__ == "__main__":
    main()