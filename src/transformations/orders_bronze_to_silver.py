from __future__ import annotations
import argparse
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from src.common.spark import get_spark
from src.common.dq import run_dq
from src.common.versioning import hash_contract_json

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bronze_path", required=True)
    p.add_argument("--silver_path", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--expectations", required=True)
    p.add_argument("--run_id", required=True)
    args = p.parse_args()

    spark = get_spark("orders_bronze_to_silver")
    schema_version = hash_contract_json(args.contract)

    df = spark.read.format("delta").load(args.bronze_path)

    clean = (
        df.select(
            F.col("order_id").cast("string").alias("order_id"),
            F.col("customer_id").cast("string").alias("customer_id"),
            F.col("order_purchase_ts").cast("timestamp").alias("order_purchase_ts"),
            F.col("order_status").cast("string").alias("order_status"),
            F.col("_ingest_run_id"),
            F.col("_ingest_ts"),
        )
        .filter(F.col("order_id").isNotNull())
    )

    # Deterministic dedupe: keep latest ingest_ts per order_id
    w = Window.partitionBy("order_id").orderBy(F.col("_ingest_ts").desc())
    dedup = clean.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")

    dq = run_dq(dedup, args.expectations)
    if not dq.passed:
        raise RuntimeError(f"Silver DQ gate failed: {dq.metrics}")

    out = (
        dedup.withColumn("_schema_version", F.lit(schema_version))
             .withColumn("_silver_run_id", F.lit(args.run_id))
             .withColumn("_silver_ts", F.current_timestamp())
    )

    # For slice: overwrite is acceptable; upgrade to MERGE later
    out.write.format("delta").mode("overwrite").save(args.silver_path)

if __name__ == "__main__":
    main()
