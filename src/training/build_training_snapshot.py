from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common.spark import get_spark


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()[:16]


def _delta_exists(spark, path: str) -> bool:
    try:
        return DeltaTable.isDeltaTable(spark, path)
    except Exception:
        return False


def _single_distinct_value(df: DataFrame, column_name: str) -> str:
    values = [row[0] for row in df.select(column_name).distinct().collect()]
    if len(values) != 1:
        raise RuntimeError(
            f"Expected exactly one distinct {column_name} in training assembly, found {values}"
        )
    return str(values[0])


def _schema_hash(df: DataFrame) -> str:
    schema_obj = json.loads(df.schema.json())
    return _hash_obj(schema_obj)


def _assert_training_snapshot_quality(df: DataFrame) -> dict[str, int]:
    null_rows = df.filter(
        F.col("customer_id").isNull()
        | F.col("as_of_date").isNull()
        | F.col("churn_label").isNull()
        | F.col("recency_days").isNull()
        | F.col("orders_30d").isNull()
        | F.col("orders_90d").isNull()
        | F.col("lifetime_orders").isNull()
        | F.col("customer_tenure_days").isNull()
        | F.col("avg_days_between_orders").isNull()
        | F.col("_feature_version").isNull()
        | F.col("_label_version").isNull()
    ).count()

    duplicate_keys = (
        df.groupBy("customer_id", "as_of_date")
        .count()
        .filter(F.col("count") > 1)
        .count()
    )

    invalid_labels = df.filter(~F.col("churn_label").isin([0, 1])).count()

    metrics = {
        "null_rows": null_rows,
        "duplicate_keys": duplicate_keys,
        "invalid_labels": invalid_labels,
    }

    if any(value > 0 for value in metrics.values()):
        raise RuntimeError(f"Training snapshot quality checks failed: {metrics}")

    return metrics


def _write_metadata(
    metadata_path: str | None,
    *,
    data_snapshot_id: str,
    row_count: int,
    as_of_date_min: str,
    as_of_date_max: str,
    feature_version: str,
    label_version: str,
    payload_schema_hash: str,
    quality_metrics: dict[str, int],
) -> None:
    if not metadata_path:
        return

    payload = {
        "data_snapshot_id": data_snapshot_id,
        "row_count": row_count,
        "as_of_date_min": as_of_date_min,
        "as_of_date_max": as_of_date_max,
        "feature_version": feature_version,
        "label_version": label_version,
        "payload_schema_hash": payload_schema_hash,
        "quality_metrics": quality_metrics,
    }

    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--training_snapshot_path", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional JSON output path for training snapshot metadata.",
    )
    parser.add_argument("--as_of_start_date", default=None)
    parser.add_argument("--as_of_end_date", default=None)
    args = parser.parse_args()

    spark = get_spark("build_training_snapshot")

    gold = spark.read.format("delta").load(args.gold_path)
    labels = spark.read.format("delta").load(args.labels_path)

    if args.as_of_start_date:
        gold = gold.filter(F.col("as_of_date") >= F.to_date(F.lit(args.as_of_start_date)))
        labels = labels.filter(F.col("as_of_date") >= F.to_date(F.lit(args.as_of_start_date)))

    if args.as_of_end_date:
        gold = gold.filter(F.col("as_of_date") <= F.to_date(F.lit(args.as_of_end_date)))
        labels = labels.filter(F.col("as_of_date") <= F.to_date(F.lit(args.as_of_end_date)))

    joined = (
        gold.alias("g")
        .join(
            labels.alias("l"),
            on=["customer_id", "as_of_date"],
            how="inner",
        )
    )

    payload = joined.select(
        "customer_id",
        "as_of_date",
        "recency_days",
        "orders_30d",
        "orders_90d",
        "lifetime_orders",
        "customer_tenure_days",
        "avg_days_between_orders",
        "churn_label",
        F.col("g._snapshot_id").alias("_feature_snapshot_id"),
        F.col("g._feature_version").alias("_feature_version"),
        F.col("l._label_version").alias("_label_version"),
        F.col("l._label_horizon_days").alias("_label_horizon_days"),
    )

    quality_metrics = _assert_training_snapshot_quality(payload)
    row_count = payload.count()
    if row_count == 0:
        raise RuntimeError("Training snapshot assembly produced zero rows.")

    feature_version = _single_distinct_value(payload, "_feature_version")
    label_version = _single_distinct_value(payload, "_label_version")
    label_horizon_days = _single_distinct_value(payload, "_label_horizon_days")

    date_bounds = payload.agg(
        F.min("as_of_date").alias("as_of_date_min"),
        F.max("as_of_date").alias("as_of_date_max"),
    ).collect()[0]

    as_of_date_min = str(date_bounds["as_of_date_min"])
    as_of_date_max = str(date_bounds["as_of_date_max"])

    payload_schema_hash = _schema_hash(payload)

    data_snapshot_id = _hash_obj(
        {
            "as_of_date_min": as_of_date_min,
            "as_of_date_max": as_of_date_max,
            "feature_version": feature_version,
            "label_version": label_version,
            "label_horizon_days": label_horizon_days,
            "payload_schema_hash": payload_schema_hash,
            "version": 1,
        }
    )

    training_snapshot = (
        payload.withColumn("_data_snapshot_id", F.lit(data_snapshot_id))
        .withColumn("_training_run_id", F.lit(args.run_id))
        .withColumn("_training_ts", F.current_timestamp())
        .select(
            "customer_id",
            "as_of_date",
            "recency_days",
            "orders_30d",
            "orders_90d",
            "lifetime_orders",
            "customer_tenure_days",
            "avg_days_between_orders",
            "churn_label",
            "_feature_snapshot_id",
            "_feature_version",
            "_label_version",
            "_label_horizon_days",
            "_data_snapshot_id",
            "_training_run_id",
            "_training_ts",
        )
    )

    _write_metadata(
        args.metadata_path,
        data_snapshot_id=data_snapshot_id,
        row_count=row_count,
        as_of_date_min=as_of_date_min,
        as_of_date_max=as_of_date_max,
        feature_version=feature_version,
        label_version=label_version,
        payload_schema_hash=payload_schema_hash,
        quality_metrics=quality_metrics,
    )

    if _delta_exists(spark, args.training_snapshot_path):
        target = DeltaTable.forPath(spark, args.training_snapshot_path)
        (
            target.alias("t")
            .merge(
                training_snapshot.alias("s"),
                "t.customer_id = s.customer_id "
                "AND t.as_of_date = s.as_of_date "
                "AND t._data_snapshot_id = s._data_snapshot_id",
            )
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        training_snapshot.write.format("delta").mode("overwrite").save(args.training_snapshot_path)


if __name__ == "__main__":
    main()