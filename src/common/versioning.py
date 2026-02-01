from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any

def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def hash_contract_json(path: str | Path) -> str:
    p = Path(path)
    obj = json.loads(p.read_text(encoding="utf-8"))
    return hashlib.sha256(_canonical_json(obj)).hexdigest()[:16]
