import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.features import load_preprocessor, transform_features
from src.predict import load_model, load_threshold, predict

preprocessor = load_preprocessor("models/preprocessor.joblib")
model = load_model("models/random_forest_model.joblib")
threshold = load_threshold("models/decision_threshold.json")

test_df = pd.read_csv("data/test.csv")
y_true = test_df["is_late"]

result = predict(test_df, preprocessor, model, threshold)

print("accuracy: ", accuracy_score(y_true, result["is_late_prediction"]))
print("precision:", precision_score(y_true, result["is_late_prediction"], zero_division=0))
print("recall:   ", recall_score(y_true, result["is_late_prediction"], zero_division=0))
print("f1:       ", f1_score(y_true, result["is_late_prediction"], zero_division=0))
print("roc_auc:  ", roc_auc_score(y_true, result["late_probability"]))