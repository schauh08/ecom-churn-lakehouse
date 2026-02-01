from fastapi import FastAPI
from services.api.app.routers.predict import router as predict_router

app = FastAPI(title="Ecomm Churn API")
app.include_router(predict_router)

@app.get("/health")
def health():
    return {"status": "ok"}

