import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.config import load_config
from src.features import load_preprocessor, transform_features
from src.predict import load_model, load_threshold, predict

config = load_config()

preprocessor = load_preprocessor(config["model"]["preprocessor_path"])
model = load_model(config["model"]["path"])
threshold = load_threshold(config["model"]["threshold_path"])

test_df = pd.read_csv(config["data"]["test_path"])
y_true = test_df["is_late"]

result = predict(test_df, preprocessor, model, threshold)

print("accuracy: ", accuracy_score(y_true, result["is_late_prediction"]))
print("precision:", precision_score(y_true, result["is_late_prediction"], zero_division=0))
print("recall:   ", recall_score(y_true, result["is_late_prediction"], zero_division=0))
print("f1:       ", f1_score(y_true, result["is_late_prediction"], zero_division=0))
print("roc_auc:  ", roc_auc_score(y_true, result["late_probability"]))
