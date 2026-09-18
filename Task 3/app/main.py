import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
import time
import pandas as pd
from fastapi import FastAPI, HTTPException


metrics_state = {
    "request_count": 0,
    "error_count": 0,
    "total_latency_ms": 0.0,
    "predictions_late": 0,
    "predictions_on_time": 0,
}

from src.config import load_config
from src.logging_config import setup_logging
from src.features import load_preprocessor
from src.predict import load_model, load_model_from_registry, load_threshold
from src.service import run_prediction
from src.exceptions import OrderValidationError
from app.schemas import OrderInput, BatchInput, PredictionOutput, BatchPredictionOutput

setup_logging()
logger = logging.getLogger(__name__)

config = load_config()
preprocessor = load_preprocessor(config["model"]["preprocessor_path"])
threshold = load_threshold(config["model"]["threshold_path"])

MODEL_VERSION = "unknown"
try:
    model = load_model_from_registry(
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["model_name"],
        config["mlflow"]["stage"].lower(),
    )
    MODEL_VERSION = f"{config['mlflow']['model_name']}@{config['mlflow']['stage'].lower()}"
    logger.info("Loaded model from MLflow registry | version=%s", MODEL_VERSION)
except Exception:
    logger.exception("Could not load model from MLflow registry, falling back to local file")
    model = load_model(config["model"]["path"])
    MODEL_VERSION = "local-fallback"

app = FastAPI(title="Late Delivery Prediction Service")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": config["mlflow"]["model_name"],
        "stage": config["mlflow"]["stage"],
        "model_version": MODEL_VERSION,
        "decision_threshold": threshold,
    }


@app.post("/predict", response_model=PredictionOutput)
def predict_single(order: OrderInput):
    start = time.perf_counter()
    metrics_state["request_count"] += 1

    df = pd.DataFrame([order.model_dump()])
    try:
        result = run_prediction(df, preprocessor, model, threshold, model_version=MODEL_VERSION)
    except OrderValidationError as e:
        metrics_state["error_count"] += 1
        raise HTTPException(status_code=422, detail={"message": str(e), "failed_checks": e.failed_expectations})

    metrics_state["total_latency_ms"] += (time.perf_counter() - start) * 1000

    row = result.iloc[0]
    if int(row["is_late_prediction"]) == 1:
        metrics_state["predictions_late"] += 1
    else:
        metrics_state["predictions_on_time"] += 1

    return PredictionOutput(
        is_late_prediction=int(row["is_late_prediction"]),
        late_probability=float(row["late_probability"]),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchInput):

    start = time.perf_counter()
    metrics_state["request_count"] += 1

    df = pd.DataFrame([o.model_dump() for o in batch.orders])

    try:
        result = run_prediction(
            df,
            preprocessor,
            model,
            threshold,
            model_version=MODEL_VERSION,
        )
    except OrderValidationError as e:
        metrics_state["error_count"] += 1
        raise HTTPException(
            status_code=422,
            detail={
                "message": str(e),
                "failed_checks": e.failed_expectations,
            },
        )

    metrics_state["total_latency_ms"] += (
        time.perf_counter() - start
    ) * 1000

    predictions = []

    for _, row in result.iterrows():
        prediction = int(row["is_late_prediction"])

        if prediction == 1:
            metrics_state["predictions_late"] += 1
        else:
            metrics_state["predictions_on_time"] += 1

        predictions.append(
            PredictionOutput(
                is_late_prediction=prediction,
                late_probability=float(row["late_probability"]),
                model_version=MODEL_VERSION,
            )
        )

    return BatchPredictionOutput(predictions=predictions)

@app.get("/metrics")
def metrics():
    count = metrics_state["request_count"]
    avg_latency = metrics_state["total_latency_ms"] / count if count else 0
    error_rate = metrics_state["error_count"] / count if count else 0
    return {
        "request_count": count,
        "error_count": metrics_state["error_count"],
        "error_rate": round(error_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "predictions_late": metrics_state["predictions_late"],
        "predictions_on_time": metrics_state["predictions_on_time"],
    }
