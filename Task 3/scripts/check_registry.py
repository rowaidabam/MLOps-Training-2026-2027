import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import mlflow
from src.config import load_config

config = load_config()
mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
client = mlflow.MlflowClient()

model = client.get_registered_model(config["mlflow"]["model_name"])
print(model)

for version in client.search_model_versions(f"name='{config['mlflow']['model_name']}'"):
    print("Version:", version.version, "| Aliases:", version.aliases)
