from __future__ import annotations
import argparse
from pyspark.sql import functions as F
from src.common.spark import get_spark
from src.common.versioning import hash_contract_json

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--silver_path", required=True)
    p.add_argument("--gold_path", required=True)
    p.add_argument("--contract", required=True)
    p.add_argument("--as_of_date", required=True)  # YYYY-MM-DD
    p.add_argument("--run_id", required=True)
    args = p.parse_args()

    spark = get_spark("customer_features_daily")
    feature_version = hash_contract_json(args.contract)

    as_of_date = F.to_date(F.lit(args.as_of_date))
    as_of_ts = F.to_timestamp(F.concat(F.lit(args.as_of_date), F.lit(" 23:59:59")))

    orders = spark.read.format("delta").load(args.silver_path)
    o = orders.filter(F.col("order_purchase_ts") <= as_of_ts)  # PIT guard

    last_order = o.groupBy("customer_id").agg(F.max("order_purchase_ts").alias("last_order_ts"))
    recency = last_order.withColumn("recency_days", F.datediff(as_of_date, F.to_date("last_order_ts")))

    orders_30d = (
        o.filter(F.col("order_purchase_ts") >= F.date_sub(as_of_ts, 30))
         .groupBy("customer_id")
         .agg(F.countDistinct("order_id").alias("orders_30d"))
    )

    features = (
        recency.select("customer_id", "recency_days")
               .join(orders_30d, on="customer_id", how="left")
               .fillna({"orders_30d": 0})
               .withColumn("as_of_date", as_of_date)
               .withColumn("_feature_version", F.lit(feature_version))
               .withColumn("_gold_run_id", F.lit(args.run_id))
               .withColumn("_gold_ts", F.current_timestamp())
    )

    features.write.format("delta").mode("append").save(args.gold_path)

if __name__ == "__main__":
    main()
