import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import mlflow
import mlflow.sklearn

from src.config import load_config
from src.features import load_preprocessor
from src.predict import load_model, load_threshold

config = load_config()

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
mlflow.set_experiment("late-delivery-classifier")

model = load_model(config["model"]["path"])
threshold = load_threshold(config["model"]["threshold_path"])

with open("models/results_summary.json") as f:
    results = json.load(f)

metrics = results["final_test_results"]
params = model.get_params()

with mlflow.start_run(run_name="random-forest-notebook-06") as run:
    mlflow.log_params(params)
    mlflow.log_param("decision_threshold", threshold)

    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1", metrics["f1"])
    mlflow.log_metric("roc_auc", metrics["roc_auc"])

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=config["mlflow"]["model_name"],
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("Run ID:", run.info.run_id)
    print("Registered model URI:", model_info.model_uri)