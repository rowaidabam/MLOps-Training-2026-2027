import os
from pathlib import Path
import yaml


def load_config(config_path="config/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    if "MLFLOW_TRACKING_URI" in os.environ:
        config["mlflow"]["tracking_uri"] = os.environ["MLFLOW_TRACKING_URI"]

    return config
