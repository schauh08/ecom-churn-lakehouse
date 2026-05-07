from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from delta.tables import DeltaTable
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.common.spark import get_spark
from src.common.pipeline_logging import get_pipeline_logger, log_pipeline_event



INVALID_LABEL_STATUSES = {"canceled", "unavailable"}


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()[:16]


def _label_policy_version(label_horizon_days: int) -> str:
    policy = {
        "entity": "customer_daily_snapshot",
        "label_name": "churn_label",
        "label_horizon_days": label_horizon_days,
        "positive_class": "no valid future order in horizon",
        "valid_future_order_invalid_statuses": sorted(INVALID_LABEL_STATUSES),
        "window": "(as_of_date, as_of_date + horizon]",
        "version": 1,
    }
    return _hash_obj(policy)


def _delta_exists(spark, path: str) -> bool:
    try:
        return DeltaTable.isDeltaTable(spark, path)
    except Exception:
        return False


def _assert_label_quality(df: DataFrame) -> dict[str, int]:
    null_rows = df.filter(
        F.col("customer_id").isNull()
        | F.col("as_of_date").isNull()
        | F.col("churn_label").isNull()
        | F.col("_label_horizon_days").isNull()
        | F.col("_label_version").isNull()
        | F.col("_labels_run_id").isNull()
        | F.col("_labels_ts").isNull()
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
        raise RuntimeError(f"Label quality checks failed: {metrics}")

    return metrics


def _write_metadata(
    metadata_path: str | None,
    *,
    as_of_date: str,
    dataset_end_date: str,
    label_horizon_days: int,
    label_version: str,
    run_id: str,
    row_count: int,
    positives: int,
    negatives: int,
    quality_metrics: dict[str, int],
) -> None:
    if not metadata_path:
        return

    payload = {
        "as_of_date": as_of_date,
        "dataset_end_date": dataset_end_date,
        "label_horizon_days": label_horizon_days,
        "label_version": label_version,
        "labels_run_id": run_id,
        "row_count": row_count,
        "positive_rows": positives,
        "negative_rows": negatives,
        "quality_metrics": quality_metrics,
    }

    path = Path(metadata_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silver_path", required=True)
    parser.add_argument("--labels_path", required=True)
    parser.add_argument("--as_of_date", required=True)  # YYYY-MM-DD
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--label_horizon_days", type=int, default=60)
    parser.add_argument(
        "--metadata_path",
        default=None,
        help="Optional JSON output path for label metadata.",
    )
    args = parser.parse_args()

    logger = get_pipeline_logger("pipeline.labels")
    log_pipeline_event(
        logger,
        "started",
        run_id=args.run_id,
        silver_path=args.silver_path,
        labels_path=args.labels_path,
        as_of_date=args.as_of_date,
        label_horizon_days=args.label_horizon_days,
    )

    try:
        spark = get_spark("labels")
        as_of_date_py = date.fromisoformat(args.as_of_date)
        label_version = _label_policy_version(args.label_horizon_days)

        silver = (
            spark.read.format("delta").load(args.silver_path)
            .select("customer_id", "order_id", "order_purchase_ts", "order_status")
            .withColumn("order_date", F.to_date("order_purchase_ts"))
        )

        dataset_end_date = silver.select(
            F.max("order_date").alias("dataset_end_date")
        ).collect()[0]["dataset_end_date"]

        if dataset_end_date is None:
            raise RuntimeError("Silver orders table is empty; cannot generate labels.")

        if as_of_date_py + timedelta(days=args.label_horizon_days) > dataset_end_date:
            raise RuntimeError(
                "Requested as_of_date is not training-eligible because the full future label window "
                f"is not observable. as_of_date={args.as_of_date}, "
                f"label_horizon_days={args.label_horizon_days}, dataset_end_date={dataset_end_date}"
            )

        as_of_date_col = F.to_date(F.lit(args.as_of_date))
        label_window_end = F.date_add(as_of_date_col, args.label_horizon_days)

        eligible_customers = (
            silver.filter(F.col("order_date") <= as_of_date_col)
            .select("customer_id")
            .distinct()
        )

        valid_future_orders = (
            silver.filter(
                (F.col("order_date") > as_of_date_col)
                & (F.col("order_date") <= label_window_end)
                & (~F.col("order_status").isin(sorted(INVALID_LABEL_STATUSES)))
            )
            .select(F.col("customer_id").alias("future_customer_id"))
            .distinct()
        )

        labels = (
            eligible_customers.join(
                valid_future_orders,
                eligible_customers["customer_id"] == valid_future_orders["future_customer_id"],
                how="left",
            )
            .withColumn(
                "churn_label",
                F.when(F.col("future_customer_id").isNull(), F.lit(1)).otherwise(F.lit(0)),
            )
            .drop("future_customer_id")
            .withColumn("as_of_date", as_of_date_col)
            .withColumn("_label_horizon_days", F.lit(args.label_horizon_days))
            .withColumn("_label_version", F.lit(label_version))
            .withColumn("_labels_run_id", F.lit(args.run_id))
            .withColumn("_labels_ts", F.current_timestamp())
            .select(
                "customer_id",
                "as_of_date",
                "churn_label",
                "_label_horizon_days",
                "_label_version",
                "_labels_run_id",
                "_labels_ts",
            )
        )

        quality_metrics = _assert_label_quality(labels)
        row_count = labels.count()
        positives = labels.filter(F.col("churn_label") == 1).count()
        negatives = labels.filter(F.col("churn_label") == 0).count()

        _write_metadata(
            args.metadata_path,
            as_of_date=args.as_of_date,
            dataset_end_date=str(dataset_end_date),
            label_horizon_days=args.label_horizon_days,
            label_version=label_version,
            run_id=args.run_id,
            row_count=row_count,
            positives=positives,
            negatives=negatives,
            quality_metrics=quality_metrics,
        )

        if _delta_exists(spark, args.labels_path):
            target = DeltaTable.forPath(spark, args.labels_path)
            (
                target.alias("t")
                .merge(
                    labels.alias("s"),
                    "t.customer_id = s.customer_id AND t.as_of_date = s.as_of_date",
                )
                .whenMatchedUpdateAll()
                .whenNotMatchedInsertAll()
                .execute()
            )
            publish_mode = "merge"
        else:
            labels.write.format("delta").mode("overwrite").save(args.labels_path)
            publish_mode = "initial_overwrite"

        log_pipeline_event(
            logger,
            "completed",
            run_id=args.run_id,
            labels_path=args.labels_path,
            as_of_date=args.as_of_date,
            dataset_end_date=str(dataset_end_date),
            label_horizon_days=args.label_horizon_days,
            label_version=label_version,
            row_count=row_count,
            positives=positives,
            negatives=negatives,
            quality_metrics=quality_metrics,
            publish_mode=publish_mode,
            metadata_path=args.metadata_path,
        )

    except Exception as exc:
        log_pipeline_event(
            logger,
            "failed",
            run_id=args.run_id,
            silver_path=args.silver_path,
            labels_path=args.labels_path,
            as_of_date=args.as_of_date,
            label_horizon_days=args.label_horizon_days,
            error=str(exc),
        )
        raise

if __name__ == "__main__":
    main()