from typing import List
from pydantic import BaseModel


class OrderInput(BaseModel):
    order_purchase_timestamp: str
    order_approved_at: str
    order_estimated_delivery_date: str
    customer_state: str
    state: str
    latitude: float
    longitude: float
    item_count: float
    unique_products: float
    unique_sellers: float
    total_item_price: float
    mean_item_price: float
    max_item_price: float
    total_freight_value: float
    mean_freight_value: float
    mean_product_weight_g: float
    product_category_count: float
    seller_state_count: float
    payment_count: float
    total_payment_value: float
    mean_payment_value: float
    max_payment_installments: float


class BatchInput(BaseModel):
    orders: List[OrderInput]


class PredictionOutput(BaseModel):
    is_late_prediction: int
    late_probability: float
    model_version: str


class BatchPredictionOutput(BaseModel):
    predictions: List[PredictionOutput]