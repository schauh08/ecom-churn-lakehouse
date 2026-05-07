from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    customer_id: str = Field(..., min_length=1, max_length=128)

    @field_validator("customer_id")
    @classmethod
    def validate_customer_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("customer_id must not be blank.")
        return normalized


class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: Literal[0, 1]
    model_version: str
    feature_version: str
    request_id: str


class ErrorResponse(BaseModel):
    detail: str
    request_id: str
    errors: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    checks: dict[str, bool]
    model_version: str | None = None
    feature_version: str | None = None
    approved_model_version: str | None = None


class VersionResponse(BaseModel):
    api_name: str
    model_version: str | None = None
    feature_version: str | None = None
    approved_model_version: str | None = None
