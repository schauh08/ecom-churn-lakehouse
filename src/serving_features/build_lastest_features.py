from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from src.common.spark import get_spark


REQUIRED_COLUMNS = [
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
]


def _assert_quality(df: DataFrame) -> dict[str, int]:
    null_rows = df.filter(
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
    ).count()

    duplicate_customers = (
        df.groupBy("customer_id")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    metrics = {
        "null_rows": null_rows,
        "duplicate_customers": duplicate_customers,
    }

    if any(value > 0 for value in metrics.values()):
        raise RuntimeError(f"Latest serving-features quality checks failed: {metrics}")

    return metrics


def _write_manifest(
    path: str | None,
    *,
    latest_features_path: str,
    row_count: int,
    as_of_date_max: str | None,
    feature_versions: list[str],
    quality_metrics: dict[str, int],
    run_id: str,
) -> None:
    if not path:
        return

    payload = {
        "latest_features_path": latest_features_path,
        "row_count": row_count,
        "as_of_date_max": as_of_date_max,
        "feature_versions": feature_versions,
        "quality_metrics": quality_metrics,
        "run_id": run_id,
    }

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--latest_features_path", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--manifest_path", default=None)
    args = parser.parse_args()

    spark = get_spark("build_latest_features")

    gold = spark.read.format("delta").load(args.gold_path).select(*REQUIRED_COLUMNS)

    latest_window = Window.partitionBy("customer_id").orderBy(
        F.col("as_of_date").desc(),
        F.col("_gold_ts").desc_nulls_last(),
        F.col("_snapshot_id").desc_nulls_last(),
    )

    latest = (
        gold.withColumn("_rn", F.row_number().over(latest_window))
        .filter(F.col("_rn") == 1)
        .drop("_rn")
    )

    quality_metrics = _assert_quality(latest)
    row_count = latest.count()

    as_of_date_max_row = latest.agg(F.max("as_of_date").alias("as_of_date_max")).collect()[0]
    as_of_date_max = str(as_of_date_max_row["as_of_date_max"]) if as_of_date_max_row["as_of_date_max"] else None

    feature_versions = sorted(
        [
            str(row[0])
            for row in latest.select("_feature_version").distinct().collect()
            if row[0] is not None
        ]
    )

    latest.write.mode("overwrite").parquet(args.latest_features_path)

    _write_manifest(
        args.manifest_path,
        latest_features_path=args.latest_features_path,
        row_count=row_count,
        as_of_date_max=as_of_date_max,
        feature_versions=feature_versions,
        quality_metrics=quality_metrics,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()