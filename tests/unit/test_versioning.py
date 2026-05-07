from __future__ import annotations

import json

from src.common.versioning import hash_contract_json, stable_hash_hex


def test_stable_hash_hex_is_order_insensitive() -> None:
    left = {"b": 2, "a": 1, "nested": {"y": 2, "x": 1}}
    right = {"nested": {"x": 1, "y": 2}, "a": 1, "b": 2}

    assert stable_hash_hex(left) == stable_hash_hex(right)


def test_hash_contract_json_changes_when_contract_changes(tmp_path) -> None:
    first = tmp_path / "contract_a.json"
    second = tmp_path / "contract_b.json"

    first.write_text(json.dumps({"name": "orders", "version": 1}), encoding="utf-8")
    second.write_text(json.dumps({"name": "orders", "version": 2}), encoding="utf-8")

    assert hash_contract_json(first) != hash_contract_json(second)