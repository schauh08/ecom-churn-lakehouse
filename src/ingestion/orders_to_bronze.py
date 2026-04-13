from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T

from src.common.spark import get_spark


BRONZE_SOURCE_SCHEMA: list[tuple[str, str]] = [
    ("order_id", "string"),
    ("customer_id", "string"),
    ("order_status", "string"),
    ("order_purchase_timestamp", "string"),
    ("order_approved_at", "string"),
    ("order_delivered_carrier_date", "string"),
    ("order_delivered_customer_date", "string"),
    ("order_estimated_delivery_date", "string"),
]


def _canonical_json(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _spark_type_name(dtype: T.DataType) -> str:
    simple = dtype.simpleString()
    mapping = {
        "string": "string",
        "timestamp": "timestamp",
        "date": "date",
        "bigint": "long",
        "long": "long",
        "int": "int",
        "double": "double",
        "float": "float",
        "boolean": "boolean",
    }
    return mapping.get(simple, simple)


def _validate_raw_schema(df: DataFrame) -> None:
    actual = {field.name: _spark_type_name(field.dataType) for field in df.schema.fields}

    expected_cols = [name for name, _ in BRONZE_SOURCE_SCHEMA]
    missing = [name for name in expected_cols if name not in actual]
    if missing:
        raise ValueError(f"Missing required Bronze source columns: {missing}")

    mismatches: list[str] = []
    for name, expected_type in BRONZE_SOURCE_SCHEMA:
        actual_type = actual[name]
        if actual_type != expected_type:
            mismatches.append(f"{name}: expected {expected_type}, got {actual_type}")

    if mismatches:
        raise ValueError("Bronze schema contract type mismatch: " + "; ".join(mismatches))


def _hash_contract_schema() -> str:
    material = "|".join(f"{name}:{dtype}" for name, dtype in BRONZE_SOURCE_SCHEMA)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _to_local_path(path_str: str) -> Path | None:
    """
    Convert a Spark input file URI like file:///... into a local Path when possible.
    Returns None for non-local URIs.
    """
    parsed = urlparse(path_str)
    if parsed.scheme in ("", "file"):
        return Path(parsed.path)
    return None


def _build_source_fingerprint(input_files: list[str]) -> str:
    """
    Build a stable fingerprint for the current ingest batch.
    For local files, include path + size + mtime_ns.
    For non-local files, fall back to file URI strings.
    """
    payload: list[dict[str, object]] = []

    for file_uri in sorted(input_files):
        local_path = _to_local_path(file_uri)
        if local_path is not None and local_path.exists():
            stat = local_path.stat()
            payload.append(
                {
                    "path": str(local_path.resolve()),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        else:
            payload.append({"path": file_uri})

    return hashlib.sha256(_canonical_json(payload)).hexdigest()[:16]


def _delta_path_exists(path_str: str) -> bool:
    return Path(path_str).exists() and any(Path(path_str).rglob("*"))


def _audit_schema() -> T.StructType:
    return T.StructType(
        [
            T.StructField("dataset", T.StringType(), False),
            T.StructField("run_id", T.StringType(), False),
            T.StructField("status", T.StringType(), False),
            T.StructField("source_fingerprint", T.StringType(), False),
            T.StructField("source_path", T.StringType(), False),
            T.StructField("source_file_count", T.IntegerType(), False),
            T.StructField("row_count", T.LongType(), False),
            T.StructField("schema_hash", T.StringType(), False),
            T.StructField("ingest_ts", T.TimestampType(), False),
            T.StructField("bronze_path", T.StringType(), False),
            T.StructField("message", T.StringType(), True),
        ]
    )


def _write_audit_record(
    spark: SparkSession,
    audit_path: str,
    *,
    dataset: str,
    run_id: str,
    status: str,
    source_fingerprint: str,
    source_path: str,
    source_file_count: int,
    row_count: int,
    schema_hash: str,
    ingest_ts: datetime,
    bronze_path: str,
    message: str | None = None,
) -> None:
    record = [
        {
            "dataset": dataset,
            "run_id": run_id,
            "status": status,
            "source_fingerprint": source_fingerprint,
            "source_path": source_path,
            "source_file_count": source_file_count,
            "row_count": row_count,
            "schema_hash": schema_hash,
            "ingest_ts": ingest_ts,
            "bronze_path": bronze_path,
            "message": message,
        }
    ]

    (
        spark.createDataFrame(record, schema=_audit_schema())
        .write.format("delta")
        .mode("append")
        .save(audit_path)
    )


def _already_ingested(
    spark: SparkSession,
    audit_path: str,
    *,
    dataset: str,
    source_fingerprint: str,
) -> bool:
    if not _delta_path_exists(audit_path):
        return False

    audit_df = spark.read.format("delta").load(audit_path)

    existing = (
        audit_df.filter(F.col("dataset") == dataset)
        .filter(F.col("source_fingerprint") == source_fingerprint)
        .filter(F.col("status") == "success")
        .limit(1)
        .count()
    )
    return existing > 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw parquet file or directory")
    parser.add_argument("--bronze_path", required=True, help="Delta output path for Bronze orders")
    parser.add_argument(
        "--audit_path",
        required=False,
        help="Delta path for Bronze audit log; defaults to <bronze_path>_audit",
    )
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--dataset", default="orders")
    args = parser.parse_args()

    audit_path = args.audit_path or f"{args.bronze_path}_audit"
    ingest_ts = datetime.now(timezone.utc).replace(tzinfo=None)

    spark = get_spark("orders_to_bronze")

    # 1) Read raw parquet exactly as-is.
    raw_df = spark.read.parquet(args.input)

    input_files = sorted(raw_df.inputFiles())
    if not input_files:
        raise ValueError(f"No input files found under: {args.input}")

    # 2) Compute lineage + validate raw schema against the Bronze contract.
    source_fingerprint = _build_source_fingerprint(input_files)
    _validate_raw_schema(raw_df)
    schema_hash = _hash_contract_schema()

    # 3) Idempotency: do not ingest the exact same batch twice.
    if _already_ingested(
        spark,
        audit_path,
        dataset=args.dataset,
        source_fingerprint=source_fingerprint,
    ):
        _write_audit_record(
            spark,
            audit_path,
            dataset=args.dataset,
            run_id=args.run_id,
            status="skipped_already_ingested",
            source_fingerprint=source_fingerprint,
            source_path=args.input,
            source_file_count=len(input_files),
            row_count=0,
            schema_hash=schema_hash,
            ingest_ts=ingest_ts,
            bronze_path=args.bronze_path,
            message="Input fingerprint already exists in Bronze audit log.",
        )
        return

    row_count = raw_df.count()

    # 4) Stamp row-level Bronze metadata.
    out_df = (
        raw_df.withColumn("run_id", F.lit(args.run_id))
        .withColumn("ingest_ts", F.lit(ingest_ts).cast("timestamp"))
        .withColumn("ingest_date", F.to_date(F.col("ingest_ts")))
        .withColumn("source_file", F.input_file_name())
        .withColumn("source_fingerprint", F.lit(source_fingerprint))
        .withColumn("row_count", F.lit(row_count).cast("long"))
        .withColumn("schema_hash", F.lit(schema_hash))
    )

    # 5) Append-only Bronze write, partitioned by ingest date.
    (
        out_df.write.format("delta")
        .mode("append")
        .partitionBy("ingest_date")
        .save(args.bronze_path)
    )

    # 6) Write one success audit record for the batch.
    _write_audit_record(
        spark,
        audit_path,
        dataset=args.dataset,
        run_id=args.run_id,
        status="success",
        source_fingerprint=source_fingerprint,
        source_path=args.input,
        source_file_count=len(input_files),
        row_count=row_count,
        schema_hash=schema_hash,
        ingest_ts=ingest_ts,
        bronze_path=args.bronze_path,
        message=None,
    )


if __name__ == "__main__":
    main()