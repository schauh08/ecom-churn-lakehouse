from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from services.api.app.observability.logging import (
    RequestContextMiddleware,
    increment_counter,
    log_event,
    setup_logging,
    snapshot_counters,
)
from services.api.app.routers.predict import (
    get_feature_store,
    get_model_store,
    router as predict_router,
)
from services.api.app.schemas.predict import (
    ErrorResponse,
    HealthResponse,
    ReadinessResponse,
    VersionResponse,
)


setup_logging()
logger = logging.getLogger("api.main")

app = FastAPI(title="Ecomm Churn API")
app.add_middleware(RequestContextMiddleware)
app.include_router(predict_router)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    request_id = getattr(request.state, "request_id", "unknown")
    increment_counter("http_validation_errors_total")
    log_event(
        logger,
        logging.WARNING,
        "request_validation_failed",
        request_id=request_id,
        path=request.url.path,
        errors=exc.errors(),
    )
    payload = ErrorResponse(
        detail="Invalid request payload.",
        request_id=request_id,
        errors=exc.errors(),
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    request_id = getattr(request.state, "request_id", "unknown")
    increment_counter(f"http_exception_{exc.status_code}_total")
    log_event(
        logger,
        logging.WARNING,
        "http_exception",
        request_id=request_id,
        path=request.url.path,
        status_code=exc.status_code,
        detail=exc.detail,
    )
    payload = ErrorResponse(
        detail=str(exc.detail),
        request_id=request_id,
        errors=None,
    )
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadinessResponse)
def ready():
    model_store = get_model_store()
    feature_store = get_feature_store()

    model_ready = model_store.is_ready()
    features_ready = feature_store.is_ready()
    all_ready = model_ready and features_ready

    response = ReadinessResponse(
        status="ready" if all_ready else "not_ready",
        checks={
            "model_store_ready": model_ready,
            "feature_store_ready": features_ready,
            "observability_ready": True,
        },
        model_version=model_store.model_version() if model_ready else None,
        feature_version=model_store.feature_version() if model_ready else None,
        approved_model_version=model_store.approved_model_version() if model_ready else None,
    )

    if not all_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=response.model_dump(),
        )

    return response


@app.get("/version", response_model=VersionResponse)
def version() -> VersionResponse:
    model_store = get_model_store()

    if not model_store.is_ready():
        return VersionResponse(api_name="Ecomm Churn API")

    return VersionResponse(
        api_name="Ecomm Churn API",
        model_version=model_store.model_version(),
        feature_version=model_store.feature_version(),
        approved_model_version=model_store.approved_model_version(),
    )


@app.get("/metrics")
def metrics():
    return snapshot_counters()