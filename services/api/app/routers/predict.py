from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Request, status

from services.api.app.auth.api_key import require_api_key
from services.api.app.feature_client.local_latest_features import LocalLatestFeaturesStore
from services.api.app.inference.model_loader import LocalModelStore
from services.api.app.observability.logging import hash_identifier, log_event
from services.api.app.schemas.predict import ErrorResponse, PredictRequest, PredictResponse


logger = logging.getLogger("api.predict")
router = APIRouter()


@lru_cache(maxsize=1)
def get_feature_store() -> LocalLatestFeaturesStore:
    features_path = os.getenv(
        "LATEST_FEATURES_PATH",
        "artifacts/serving/latest_features",
    )
    return LocalLatestFeaturesStore(features_path)


@lru_cache(maxsize=1)
def get_model_store() -> LocalModelStore:
    model_path = os.getenv(
        "MODEL_PATH",
        "artifacts/models/ecomm_churn_baseline.pkl",
    )
    model_meta_path = os.getenv(
        "MODEL_META_PATH",
        "artifacts/models/model_meta.json",
    )
    approved_path = os.getenv(
        "APPROVED_MODEL_VERSION_PATH",
        "artifacts/models/approved_model_version.json",
    )
    return LocalModelStore(
        model_path=model_path,
        model_meta_path=model_meta_path,
        approved_model_version_path=approved_path,
    )


@router.post(
    "/v1/churn/predict",
    response_model=PredictResponse,
    responses={
        401: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
def predict(
    req: PredictRequest,
    request: Request,
    _: str = Depends(require_api_key),
) -> PredictResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    customer_id_hash = hash_identifier(req.customer_id)

    feature_store = get_feature_store()
    model_store = get_model_store()

    if not feature_store.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Latest feature store is not ready.",
        )

    if not model_store.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model store is not ready.",
        )

    feature_record = feature_store.get(req.customer_id)
    if feature_record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No serving features found for customer_id.",
        )

    expected_feature_version = model_store.feature_version()
    if expected_feature_version and feature_record.feature_version != expected_feature_version:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Serving feature version does not match approved model feature version. "
                f"serving={feature_record.feature_version}, model={expected_feature_version}"
            ),
        )

    probability = model_store.predict_probability(feature_record.values)
    churn_label = 1 if probability >= 0.5 else 0

    log_event(
        logger,
        logging.INFO,
        "prediction_scored",
        request_id=request_id,
        customer_id_hash=customer_id_hash,
        model_version=model_store.model_version(),
        feature_version=feature_record.feature_version,
        as_of_date=feature_record.as_of_date,
        churn_probability=round(probability, 6),
        churn_label=churn_label,
    )

    return PredictResponse(
        customer_id=req.customer_id,
        churn_probability=probability,
        churn_label=churn_label,
        model_version=str(model_store.model_version()),
        feature_version=feature_record.feature_version,
        request_id=request_id,
    )