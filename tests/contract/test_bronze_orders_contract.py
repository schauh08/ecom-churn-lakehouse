import json
from pathlib import Path

from pyspark.sql.types import (
    StructField,
    StructType,
    StringType,
    TimestampType,
    DateType,
    LongType,
)

CONTRACT_PATH = Path("data/contracts/bronze/orders.v1.json")


def load_contract():
    return json.loads(CONTRACT_PATH.read_text())


def spark_type_name(dtype) -> str:
    mapping = {
        "string": "string",
        "timestamp": "timestamp",
        "date": "date",
        "bigint": "long",
        "long": "long",
    }
    return mapping.get(dtype.simpleString(), dtype.simpleString())


def test_bronze_orders_contract_matches_expected_schema(spark):
    contract = load_contract()

    schema = StructType([
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField("order_purchase_timestamp", StringType(), True),
        StructField("order_approved_at", StringType(), True),
        StructField("order_delivered_carrier_date", StringType(), True),
        StructField("order_delivered_customer_date", StringType(), True),
        StructField("order_estimated_delivery_date", StringType(), True),
        StructField("run_id", StringType(), False),
        StructField("ingest_ts", TimestampType(), False),
        StructField("ingest_date", DateType(), False),
        StructField("source_file", StringType(), False),
        StructField("source_fingerprint", StringType(), False),
        StructField("row_count", LongType(), False),
        StructField("schema_hash", StringType(), False),
    ])

    actual_cols = {field.name for field in schema.fields}
    actual_types = {
        field.name: spark_type_name(field.dataType)
        for field in schema.fields
    }

    required = set(contract["required_columns"])
    assert required.issubset(actual_cols), (
        f"Missing required columns: {sorted(required - actual_cols)}"
    )

    for col, spec in contract["columns"].items():
        assert col in actual_types, f"{col} missing from schema"
        assert actual_types[col] == spec["type"], (
            f"{col}: expected {spec['type']}, got {actual_types[col]}"
        )

    if not contract["allow_extra_columns"]:
        assert actual_cols == set(contract["columns"].keys()), (
            f"Unexpected extra columns: {sorted(actual_cols - set(contract['columns'].keys()))}"
        )