import json
import pytest
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.features import load_preprocessor
from src.predict import load_model, load_threshold, predict


def test_pipeline_matches_notebook_test_results():
    preprocessor = load_preprocessor("models/preprocessor.joblib")
    model = load_model("models/random_forest_model.joblib")
    threshold = load_threshold("models/decision_threshold.json")

    test_df = pd.read_csv("data/test.csv")
    y_true = test_df["is_late"]

    result = predict(test_df, preprocessor, model, threshold)

    with open("models/results_summary.json") as f:
        expected = json.load(f)["final_test_results"]

    assert accuracy_score(y_true, result["is_late_prediction"]) == pytest.approx(expected["accuracy"], abs=1e-6)
    assert precision_score(y_true, result["is_late_prediction"], zero_division=0) == pytest.approx(expected["precision"], abs=1e-6)
    assert recall_score(y_true, result["is_late_prediction"], zero_division=0) == pytest.approx(expected["recall"], abs=1e-6)
    assert f1_score(y_true, result["is_late_prediction"], zero_division=0) == pytest.approx(expected["f1"], abs=1e-6)
    assert roc_auc_score(y_true, result["late_probability"]) == pytest.approx(expected["roc_auc"], abs=1e-6)