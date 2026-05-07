from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.app.main import app


def test_predict_validation_failure_returns_clean_error(monkeypatch) -> None:
    monkeypatch.setenv("API_KEY", "test-api-key")
    client = TestClient(app)

    response = client.post(
        "/v1/churn/predict",
        headers={"X-API-Key": "test-api-key"},
        json={"customer_id": "   "},
    )

    assert response.status_code == 422
    body = response.json()

    assert body["detail"] == "Invalid request payload."
    assert body["request_id"]
    assert isinstance(body["errors"], list)