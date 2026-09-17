import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "model_version" in response.json()


def test_predict_valid_order():
    order = {
        "order_purchase_timestamp": "2018-01-01T10:00:00",
        "order_approved_at": "2018-01-01T12:00:00",
        "order_estimated_delivery_date": "2018-01-15T00:00:00",
        "customer_state": "SP",
        "state": "SP",
        "latitude": -23.5,
        "longitude": -46.6,
        "item_count": 1,
        "unique_products": 1,
        "unique_sellers": 1,
        "total_item_price": 100.0,
        "mean_item_price": 100.0,
        "max_item_price": 100.0,
        "total_freight_value": 15.0,
        "mean_freight_value": 15.0,
        "mean_product_weight_g": 500.0,
        "product_category_count": 1,
        "seller_state_count": 1,
        "payment_count": 1,
        "total_payment_value": 115.0,
        "mean_payment_value": 115.0,
        "max_payment_installments": 1,
    }
    response = client.post("/predict", json=order)
    assert response.status_code == 200
    body = response.json()
    assert "is_late_prediction" in body
    assert "late_probability" in body


def test_predict_missing_field_returns_422():
    order = {"order_purchase_timestamp": "2018-01-01T10:00:00"}  # missing everything else
    response = client.post("/predict", json=order)
    assert response.status_code == 422