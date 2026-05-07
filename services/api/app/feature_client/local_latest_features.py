from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


FEATURE_COLUMNS = [
    "recency_days",
    "orders_30d",
    "orders_90d",
    "lifetime_orders",
    "customer_tenure_days",
    "avg_days_between_orders",
]


@dataclass(frozen=True)
class FeatureRecord:
    customer_id: str
    as_of_date: str
    feature_version: str
    snapshot_id: str
    values: dict[str, float]


class LocalLatestFeaturesStore:
    def __init__(self, features_path: str) -> None:
        self.features_path = features_path
        self._records: dict[str, FeatureRecord] | None = None

    def _load_if_needed(self) -> None:
        if self._records is not None:
            return

        path = Path(self.features_path)
        if not path.exists():
            raise FileNotFoundError(f"Latest features path does not exist: {self.features_path}")

        df = pd.read_parquet(self.features_path)

        required_columns = {
            "customer_id",
            "as_of_date",
            "_feature_version",
            "_snapshot_id",
            *FEATURE_COLUMNS,
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise RuntimeError(
                f"Latest features store missing columns: {sorted(missing)}. "
                f"Found: {list(df.columns)}"
            )

        records: dict[str, FeatureRecord] = {}
        for _, row in df.iterrows():
            customer_id = str(row["customer_id"])
            records[customer_id] = FeatureRecord(
                customer_id=customer_id,
                as_of_date=str(row["as_of_date"]),
                feature_version=str(row["_feature_version"]),
                snapshot_id=str(row["_snapshot_id"]),
                values={
                    "recency_days": float(row["recency_days"]),
                    "orders_30d": float(row["orders_30d"]),
                    "orders_90d": float(row["orders_90d"]),
                    "lifetime_orders": float(row["lifetime_orders"]),
                    "customer_tenure_days": float(row["customer_tenure_days"]),
                    "avg_days_between_orders": float(row["avg_days_between_orders"]),
                },
            )

        self._records = records

    def is_ready(self) -> bool:
        try:
            self._load_if_needed()
            return True
        except Exception:
            return False
    def get(self, customer_id: str) -> FeatureRecord | None:
        self._load_if_needed()
        assert self._records is not None
        return self._records.get(customer_id)

    def row_count(self) -> int:
        self._load_if_needed()
        assert self._records is not None
        return len(self._records)