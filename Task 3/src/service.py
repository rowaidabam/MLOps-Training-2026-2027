import logging
import pandas as pd

from src.validation import validate_orders
from src.predict import predict
from src.exceptions import OrderValidationError

logger = logging.getLogger(__name__)


def run_prediction(df: pd.DataFrame, preprocessor, model, threshold, model_version="unknown") -> pd.DataFrame:
    validation_result = validate_orders(df)

    if not validation_result.success:
        failed_checks = [
            f"{r.expectation_config.type} - {r.expectation_config.kwargs.get('column')}"
            for r in validation_result.results
            if not r.success
        ]
        logger.warning("Input validation failed | failed_checks=%s", failed_checks)
        raise OrderValidationError(
            "Input data failed validation checks",
            failed_expectations=failed_checks,
        )

    return predict(df, preprocessor, model, threshold, model_version=model_version)