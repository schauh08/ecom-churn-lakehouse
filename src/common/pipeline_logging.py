from __future__ import annotations

import json
import logging
from typing import Any


def get_pipeline_logger(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return logging.getLogger(name)


def log_pipeline_event(
    logger: logging.Logger,
    event: str,
    *,
    run_id: str,
    **fields: Any,
) -> None:
    payload = {
        "event": event,
        "run_id": run_id,
        **fields,
    }
    logger.info(json.dumps(payload, default=str, sort_keys=True))