import json, uuid
from fastapi import APIRouter
from services.api.app.schemas.predict import PredictRequest, PredictResponse

router = APIRouter()
MODEL_META_PATH = "artifacts/model_meta.json"

@router.post("/v1/churn/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    request_id = str(uuid.uuid4())
    meta = json.loads(open(MODEL_META_PATH, "r", encoding="utf-8").read())

    prob = 0.42
    label = 1 if prob >= 0.5 else 0

    return PredictResponse(
        customer_id=req.customer_id,
        churn_probability=prob,
        churn_label=label,
        model_version=meta["model_version"],
        feature_version=meta["feature_version"],
        request_id=request_id
    )
