from __future__ import annotations
import argparse
from pyspark.sql import functions as F
from src.common.spark import get_spark

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--bronze_path", required=True)
    p.add_argument("--run_id", required=True)
    args = p.parse_args()

    spark = get_spark("orders_to_bronze")
    df = spark.read.parquet(args.input)

    out = (
        df.withColumn("_ingest_run_id", F.lit(args.run_id))
          .withColumn("_ingest_ts", F.current_timestamp())
          .withColumn("_source_file", F.input_file_name())
    )

    out.write.format("delta").mode("append").save(args.bronze_path)

if __name__ == "__main__":
    main()
