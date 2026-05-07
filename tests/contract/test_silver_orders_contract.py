from __future__ import annotations

import json
from pathlib import Path


CONTRACT_PATH = Path("data/contracts/silver/orders.v1.json")


def test_silver_orders_contract_matches_expected_shape() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["contract_name"] == "silver.orders"
    assert contract["primary_key"] == ["order_id"]
    assert contract["allow_extra_columns"] is False

    required_columns = set(contract["required_columns"])
    declared_columns = set(contract["columns"].keys())

    assert required_columns == declared_columns
    assert contract["columns"]["order_id"]["type"] == "string"
    assert contract["columns"]["customer_id"]["type"] == "string"
    assert contract["columns"]["order_purchase_ts"]["type"] == "timestamp"
    assert contract["columns"]["order_status"]["type"] == "string"

    allowed = set(contract["allowed_values"]["order_status"])
    assert allowed == {
        "approved",
        "canceled",
        "created",
        "delivered",
        "invoiced",
        "processing",
        "shipped",
        "unavailable",
    }