from __future__ import annotations

from fastapi.testclient import TestClient

from services.api.app.main import app


def test_predict_auth_failure_returns_401() -> None:
    client = TestClient(app)

    response = client.post(
        "/v1/churn/predict",
        json={"customer_id": "cust_0001"},
    )

    assert response.status_code == 401
    body = response.json()

    assert body["detail"] == "Invalid or missing API key."
    assert body["request_id"]