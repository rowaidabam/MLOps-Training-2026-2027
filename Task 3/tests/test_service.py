import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import pandas as pd

from src.config import load_config
from src.features import load_preprocessor
from src.predict import load_model, load_threshold
from src.service import run_prediction
from src.exceptions import OrderValidationError


@pytest.fixture
def loaded_artifacts():
    config = load_config()
    preprocessor = load_preprocessor(config["model"]["preprocessor_path"])
    model = load_model(config["model"]["path"])
    threshold = load_threshold(config["model"]["threshold_path"])
    test_df = pd.read_csv(config["data"]["test_path"])
    return preprocessor, model, threshold, test_df


def test_valid_orders_pass_through(loaded_artifacts):
    preprocessor, model, threshold, test_df = loaded_artifacts
    result = run_prediction(test_df.head(5), preprocessor, model, threshold)
    assert len(result) == 5
    assert "late_probability" in result.columns


def test_missing_required_column_is_rejected(loaded_artifacts):
    preprocessor, model, threshold, test_df = loaded_artifacts
    broken_df = test_df.head(5).drop(columns=["customer_state"])

    with pytest.raises(OrderValidationError):
        run_prediction(broken_df, preprocessor, model, threshold)