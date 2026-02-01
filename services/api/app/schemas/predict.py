from pydantic import BaseModel

class PredictRequest(BaseModel):
    customer_id: str

class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_label: int
    model_version: str
    feature_version: str
    request_id: str
