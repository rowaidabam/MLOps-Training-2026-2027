import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow
from src.config import load_config

config = load_config()
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = mlflow.MlflowClient()

alias = config["mlflow"]["stage"].lower()
client.set_registered_model_alias(config["mlflow"]["model_name"], alias, "1")
print(f"Alias '{alias}' set on version 1")