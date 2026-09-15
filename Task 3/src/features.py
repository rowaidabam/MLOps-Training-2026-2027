import json
from pathlib import Path

import joblib
import pandas as pd


DATE_COLUMNS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_estimated_delivery_date",
]

COLUMNS_TO_DROP = [
    "is_late",
    "order_id",
    "customer_id",
    "customer_unique_id",
    "order_status",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
]

HIGH_CARDINALITY_LOCATION_COLS = [
    "customer_city",
    "city",
    "customer_zip_code_prefix",
    "geolocation_zip_code_prefix",
]


def create_date_features(df):
    df = df.copy()

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["purchase_year"] = df["order_purchase_timestamp"].dt.year
    df["purchase_month"] = df["order_purchase_timestamp"].dt.month
    df["purchase_dayofweek"] = df["order_purchase_timestamp"].dt.dayofweek
    df["purchase_hour"] = df["order_purchase_timestamp"].dt.hour

    df["estimated_delivery_month"] = (
        df["order_estimated_delivery_date"].dt.month
    )

    df["estimated_delivery_window_days"] = (
        df["order_estimated_delivery_date"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / (60 * 60 * 24)

    df["approval_delay_hours"] = (
        df["order_approved_at"]
        - df["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600

    df = df.drop(columns=DATE_COLUMNS)

    return df


def prepare_features(df):
    df = df.copy()

    # Remove target and columns that must not be used for prediction
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")

    # Create the date-based features
    df = create_date_features(df)

    # Remove high-cardinality location features
    df = df.drop(
        columns=HIGH_CARDINALITY_LOCATION_COLS,
        errors="ignore",
    )

    return df


def load_preprocessor(path):
    return joblib.load(path)


def transform_features(df, preprocessor):
    prepared_df = prepare_features(df)

    transformed = preprocessor.transform(prepared_df)

    return transformed