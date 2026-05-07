from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def stable_hash_hex(obj: Any, length: int = 16) -> str:
    return hashlib.sha256(canonical_json_bytes(obj)).hexdigest()[:length]


def hash_contract_json(path: str | Path) -> str:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    return stable_hash_hex(obj, length=16)