import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from src.config import load_config
from src.features import load_preprocessor
from src.predict import load_model, load_model_from_registry, load_threshold, predict


def test_registry_model_matches_local_model():
    config = load_config()
    test_df = pd.read_csv(config["data"]["test_path"]).head(5)
    preprocessor = load_preprocessor(config["model"]["preprocessor_path"])
    threshold = load_threshold(config["model"]["threshold_path"])

    local_model = load_model(config["model"]["path"])
    registry_model = load_model_from_registry(
        config["mlflow"]["tracking_uri"],
        config["mlflow"]["model_name"],
        config["mlflow"]["stage"].lower(),
    )

    local_result = predict(test_df, preprocessor, local_model, threshold)
    registry_result = predict(test_df, preprocessor, registry_model, threshold)

    assert (local_result["late_probability"].round(6) == registry_result["late_probability"].round(6)).all()