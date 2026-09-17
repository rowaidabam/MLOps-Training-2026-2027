import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging
import pandas as pd
from fastapi import FastAPI, HTTPException

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
    df = pd.DataFrame([order.model_dump()])
    try:
        result = run_prediction(df, preprocessor, model, threshold, model_version=MODEL_VERSION)
    except OrderValidationError as e:
        raise HTTPException(status_code=422, detail={"message": str(e), "failed_checks": e.failed_expectations})

    row = result.iloc[0]
    return PredictionOutput(
        is_late_prediction=int(row["is_late_prediction"]),
        late_probability=float(row["late_probability"]),
        model_version=MODEL_VERSION,
    )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
def predict_batch(batch: BatchInput):
    df = pd.DataFrame([o.model_dump() for o in batch.orders])
    try:
        result = run_prediction(df, preprocessor, model, threshold, model_version=MODEL_VERSION)
    except OrderValidationError as e:
        raise HTTPException(status_code=422, detail={"message": str(e), "failed_checks": e.failed_expectations})

    predictions = [
        PredictionOutput(
            is_late_prediction=int(row["is_late_prediction"]),
            late_probability=float(row["late_probability"]),
            model_version=MODEL_VERSION,
        )
        for _, row in result.iterrows()
    ]
    return BatchPredictionOutput(predictions=predictions)
