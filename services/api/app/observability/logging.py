from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import time
import uuid
from collections import Counter
from threading import Lock
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="",
)

_metrics = Counter()
_metrics_lock = Lock()


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_request_id() -> str:
    return _request_id_ctx.get() or ""


def hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def increment_counter(name: str, value: float = 1.0) -> None:
    with _metrics_lock:
        _metrics[name] += value


def snapshot_counters() -> dict[str, float]:
    with _metrics_lock:
        return dict(_metrics)


def log_event(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "request_id": get_request_id() or fields.pop("request_id", None),
        **fields,
    }
    logger.log(level, json.dumps(payload, default=str, sort_keys=True))


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        token = _request_id_ctx.set(request_id)
        request.state.request_id = request_id

        logger = logging.getLogger("api")
        start = time.perf_counter()
        increment_counter("http_requests_total")
        increment_counter(f"http_requests_{request.method.lower()}_total")

        log_event(
            logger,
            logging.INFO,
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception:
            increment_counter("http_exceptions_total")
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            increment_counter("http_request_latency_ms_sum", duration_ms)
            increment_counter("http_request_latency_count")
            log_event(
                logger,
                logging.ERROR,
                "request_failed_unhandled",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
            )
            _request_id_ctx.reset(token)
            raise

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        increment_counter("http_request_latency_ms_sum", duration_ms)
        increment_counter("http_request_latency_count")
        increment_counter(f"http_status_{response.status_code}_total")

        log_event(
            logger,
            logging.INFO,
            "request_finished",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )

        response.headers["X-Request-ID"] = request_id
        _request_id_ctx.reset(token)
        return response