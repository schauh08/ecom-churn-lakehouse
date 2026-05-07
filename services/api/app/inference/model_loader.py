from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


class LocalModelStore:
    def __init__(
        self,
        model_path: str,
        model_meta_path: str | None = None,
        approved_model_version_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_meta_path = model_meta_path
        self.approved_model_version_path = approved_model_version_path

        self._bundle: dict[str, Any] | None = None
        self._meta: dict[str, Any] | None = None
        self._approved: dict[str, Any] | None = None

    def _load_if_needed(self) -> None:
        if self._bundle is not None:
            return

        model_file = Path(self.model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        with model_file.open("rb") as f:
            bundle = pickle.load(f)

        required = {
            "model_name",
            "model_version",
            "feature_columns",
            "feature_version",
            "pipeline",
        }
        missing = required - set(bundle.keys())
        if missing:
            raise RuntimeError(f"Model bundle missing required keys: {sorted(missing)}")

        self._bundle = bundle

        if self.model_meta_path:
            meta_file = Path(self.model_meta_path)
            if meta_file.exists():
                self._meta = json.loads(meta_file.read_text(encoding="utf-8"))

        if self.approved_model_version_path:
            approved_file = Path(self.approved_model_version_path)
            if approved_file.exists():
                self._approved = json.loads(approved_file.read_text(encoding="utf-8"))
                approved_version = self._approved.get("approved_model_version")
                if approved_version and approved_version != bundle["model_version"]:
                    raise RuntimeError(
                        "Loaded model_version does not match approved_model_version. "
                        f"loaded={bundle['model_version']}, approved={approved_version}"
                    )

    def is_ready(self) -> bool:
        try:
            self._load_if_needed()
            return True
        except Exception:
            return False

    def model_version(self) -> str | None:
        self._load_if_needed()
        assert self._bundle is not None
        return str(self._bundle["model_version"])

    def feature_version(self) -> str | None:
        self._load_if_needed()
        assert self._bundle is not None
        return str(self._bundle["feature_version"])

    def approved_model_version(self) -> str | None:
        self._load_if_needed()
        if self._approved is None:
            return self.model_version()
        return self._approved.get("approved_model_version")

    def predict_probability(self, features: dict[str, float]) -> float:
        self._load_if_needed()
        assert self._bundle is not None

        feature_columns = list(self._bundle["feature_columns"])
        payload = {col: float(features[col]) for col in feature_columns}
        frame = pd.DataFrame([payload])

        pipeline = self._bundle["pipeline"]
        proba = pipeline.predict_proba(frame)[:, 1][0]
        return float(proba)