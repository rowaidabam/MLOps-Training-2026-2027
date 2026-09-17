import pandas as pd
import pytest
from src.features import create_date_features, prepare_features


def make_fake_order():
    return pd.DataFrame([{
        "is_late": 0,
        "order_id": "abc123",
        "customer_id": "cust1",
        "customer_unique_id": "custuniq1",
        "order_status": "delivered",
        "order_delivered_carrier_date": "2026-01-02",
        "order_delivered_customer_date": "2026-01-05",
        "order_purchase_timestamp": "2026-01-01 10:00:00",
        "order_approved_at": "2026-01-01 12:00:00",
        "order_estimated_delivery_date": "2026-01-06",
        "customer_city": "sao paulo",
        "city": "sao paulo",
        "customer_zip_code_prefix": "01001",
        "geolocation_zip_code_prefix": "01001",
        "customer_state": "SP",
        "state": "SP",
        "latitude": -23.5,
        "longitude": -46.6,
        "item_count": 2,
    }])


def test_create_date_features_adds_expected_columns():
    df = make_fake_order()
    result = create_date_features(df)

    assert "purchase_year" in result.columns
    assert "approval_delay_hours" in result.columns
    assert "order_purchase_timestamp" not in result.columns  # raw date dropped
    assert result["purchase_year"].iloc[0] == 2026
    assert result["approval_delay_hours"].iloc[0] == 2.0  # 12:00 - 10:00


def test_prepare_features_drops_leak_and_id_columns():
    df = make_fake_order()
    result = prepare_features(df)

    for col in ["is_late", "order_id", "order_status", "order_delivered_customer_date"]:
        assert col not in result.columns

    for col in ["customer_city", "city", "customer_zip_code_prefix"]:
        assert col not in result.columns

    assert "customer_state" in result.columns
    assert "item_count" in result.columns
