import json
from pathlib import Path

CONTRACT_PATH = Path("data/contracts/silver/orders.v1.json")


def test_silver_orders_contract_has_expected_primary_key_and_allowed_statuses() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    assert contract["primary_key"] == ["order_id"]
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