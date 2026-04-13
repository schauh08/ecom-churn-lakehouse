from __future__ import annotations

import argparse
import json
from pathlib import Path

from delta.tables import DeltaTable
from pyspark.sql import DataFrame, functions as F
from pyspark.sql.window import Window

from src.common.dq import run_dq, write_dq_report, write_failed_rows
from src.common.spark import get_spark
from src.common.versioning import hash_contract_json


def _load_contract(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalize_status(column_name: str):
    raw = F.lower(F.trim(F.col(column_name)))
    return (
        F.when(raw == "cancelled", F.lit("canceled"))
        .when(raw == "shipment_pending", F.lit("processing"))
        .otherwise(raw)
    )


def _delta_exists(spark, path: str) -> bool:
    try:
        return DeltaTable.isDeltaTable(spark, path)
    except Exception:
        return False


def _write_quarantine(df: DataFrame, path: str) -> None:
    if df.limit(1).count() == 0:
        return
    df.write.mode("overwrite").parquet(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bronze_path", required=True)
    parser.add_argument("--silver_path", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--expectations", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument(
        "--dq_report_path",
        default=None,
        help="Optional JSON output path for the Silver DQ report.",
    )
    parser.add_argument(
        "--quarantine_path",
        default=None,
        help="Optional base path for quarantined bad rows.",
    )
    args = parser.parse_args()

    spark = get_spark("orders_bronze_to_silver")
    contract = _load_contract(args.contract)
    schema_version = hash_contract_json(args.contract)
    allowed_statuses = contract["allowed_values"]["order_status"]

    dq_report_path = args.dq_report_path or f"{args.silver_path}_dq_reports/{args.run_id}.json"
    quarantine_path = args.quarantine_path or f"{args.silver_path}_quarantine/{args.run_id}"

    bronze_df = spark.read.format("delta").load(args.bronze_path)

    normalized = bronze_df.select(
        F.lower(F.trim(F.col("order_id"))).alias("order_id"),
        F.lower(F.trim(F.col("customer_id"))).alias("customer_id"),
        F.to_timestamp(
            F.trim(F.col("order_purchase_timestamp")),
            "yyyy-MM-dd HH:mm:ss",
        ).alias("order_purchase_ts"),
        _normalize_status("order_status").alias("order_status"),
        F.col("run_id").alias("_bronze_run_id"),
        F.col("ingest_ts").alias("_bronze_ingest_ts"),
        F.col("source_file").alias("_bronze_source_file"),
        F.col("source_fingerprint").alias("_bronze_source_fingerprint"),
        F.col("schema_hash").alias("_bronze_schema_hash"),
    )

    # Quarantine malformed or disallowed rows before publish.
    preclean_invalid = normalized.filter(
        F.col("order_id").isNull()
        | F.col("customer_id").isNull()
        | F.col("order_purchase_ts").isNull()
        | F.col("order_status").isNull()
        | (~F.col("order_status").isin(allowed_statuses))
    )
    _write_quarantine(preclean_invalid, f"{quarantine_path}/preclean_invalid")

    clean = normalized.filter(
        F.col("order_id").isNotNull()
        & F.col("customer_id").isNotNull()
        & F.col("order_purchase_ts").isNotNull()
        & F.col("order_status").isNotNull()
        & F.col("order_status").isin(allowed_statuses)
    )

    # Deterministic business-key dedupe.
    dedupe_window = Window.partitionBy("order_id").orderBy(
        F.col("order_purchase_ts").desc_nulls_last(),
        F.col("_bronze_ingest_ts").desc_nulls_last(),
        F.col("_bronze_source_file").desc_nulls_last(),
        F.col("_bronze_run_id").desc_nulls_last(),
    )
    ranked = clean.withColumn("_row_num", F.row_number().over(dedupe_window))

    duplicate_rejects = ranked.filter(F.col("_row_num") > 1).drop("_row_num")
    _write_quarantine(duplicate_rejects, f"{quarantine_path}/duplicate_rejects")

    deduped = ranked.filter(F.col("_row_num") == 1).drop("_row_num")

    silver_out = (
        deduped.select(
            "order_id",
            "customer_id",
            "order_purchase_ts",
            "order_status",
            "_bronze_run_id",
            "_bronze_ingest_ts",
            "_bronze_source_file",
            "_bronze_source_fingerprint",
            "_bronze_schema_hash",
        )
        .withColumn("_schema_version", F.lit(schema_version))
        .withColumn("_silver_run_id", F.lit(args.run_id))
        .withColumn("_silver_ts", F.current_timestamp())
    )

    dq_result = run_dq(silver_out, args.expectations)
    write_dq_report(dq_result, dq_report_path)
    write_failed_rows(dq_result, f"{quarantine_path}/dq_failed")

    if not dq_result.passed:
        raise RuntimeError(
            f"Silver DQ gate failed. See report at {dq_report_path}. Metrics={dq_result.metrics}"
        )

    # ACID-backed idempotent publish semantics.
    if _delta_exists(spark, args.silver_path):
        target = DeltaTable.forPath(spark, args.silver_path)
        (
            target.alias("t")
            .merge(silver_out.alias("s"), "t.order_id = s.order_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )
    else:
        silver_out.write.format("delta").mode("overwrite").save(args.silver_path)


if __name__ == "__main__":
    main()