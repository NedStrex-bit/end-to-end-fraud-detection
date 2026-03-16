from fastapi.testclient import TestClient

from api.app import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict() -> None:
    payload = {
        "transaction_id": "tx_smoke_001",
        "user_id": "user_smoke",
        "transaction_time": "2026-03-01T10:15:00",
        "amount": 125.50,
        "currency": "USD",
        "merchant": "Store_A",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"fraud_probability", "fraud_label", "threshold"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["fraud_label"] in {0, 1}
    assert 0.0 <= body["threshold"] <= 1.0
