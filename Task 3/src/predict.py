import json
import logging
import time

import joblib
import pandas as pd

from src.features import load_preprocessor, transform_features


logger = logging.getLogger(__name__)


def load_model(path):
    return joblib.load(path)


def load_threshold(path):
    with open(path) as f:
        return json.load(f)["threshold"]


def predict(
    df: pd.DataFrame,
    preprocessor,
    model,
    threshold,
    model_version="unknown",
) -> pd.DataFrame:
    start_time = time.perf_counter()

    logger.info(
        "Prediction request received | rows=%d | columns=%s | model_version=%s",
        len(df),
        list(df.columns),
        model_version,
    )

    try:
        X = transform_features(df, preprocessor)

        probabilities = model.predict_proba(X)[:, 1]
        predictions = (probabilities >= threshold).astype(int)

        result = pd.DataFrame({
            "is_late_prediction": predictions,
            "late_probability": probabilities,
        })

        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "Prediction completed | rows=%d | predicted_late=%d | predicted_on_time=%d | "
            "latency_ms=%.2f | model_version=%s",
            len(df),
            int(predictions.sum()),
            int((predictions == 0).sum()),
            latency_ms,
            model_version,
        )

        return result

    except Exception:
        logger.exception(
            "Prediction failed | rows=%d | model_version=%s",
            len(df),
            model_version,
        )
        raise