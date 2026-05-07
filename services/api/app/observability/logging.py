from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import time
import uuid
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


_request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id",
    default="",
)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )


def get_request_id() -> str:
    return _request_id_ctx.get() or ""


def hash_identifier(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


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

        log_event(
            logger,
            logging.INFO,
            "request_started",
            method=request.method,
            path=request.url.path,
        )

        try:
            response = await call_next(request)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 2)
            log_event(
                logger,
                logging.INFO,
                "request_finished",
                method=request.method,
                path=request.url.path,
                duration_ms=duration_ms,
            )
            _request_id_ctx.reset(token)

        response.headers["X-Request-ID"] = request_id
        return response