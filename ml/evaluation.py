
"""
STRATUM - Regression Evaluation

Reusable evaluation utilities for semiconductor
virtual metrology models.

Metrics:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R2)
- Mean Absolute Percentage Error (MAPE)
- Mean Prediction Bias
"""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression(y_true, y_pred):
    """
    Calculate regression evaluation metrics.

    Both inputs must contain finite numerical values
    and have matching lengths.

    MAPE is reported as a percentage.
    Prediction bias is defined as predicted - actual.
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(
            "Targets and predictions must be one-dimensional."
        )

    if len(y_true) != len(y_pred):
        raise ValueError(
            "Targets and predictions have different lengths."
        )

    if len(y_true) < 2:
        raise ValueError(
            "At least two observations are required."
        )

    if not np.isfinite(y_true).all():
        raise ValueError(
            "Actual values contain non-finite observations."
        )

    if not np.isfinite(y_pred).all():
        raise ValueError(
            "Predictions contain non-finite observations."
        )

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    # MAPE is undefined when an actual value is zero.
    if np.any(y_true == 0):
        mape = np.nan
    else:
        mape = np.mean(
            np.abs(
                (y_true - y_pred) / y_true
            )
        ) * 100

    bias = np.mean(y_pred - y_true)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE (%)": float(mape),
        "Bias": float(bias),
    }


def calculate_mae_improvement(
    baseline_mae,
    model_mae,
):
    """
    Calculate percentage improvement over a baseline.

    Positive value:
        Model has lower MAE than the baseline.

    Negative value:
        Model performs worse than the baseline.
    """

    if baseline_mae <= 0:
        raise ValueError(
            "Baseline MAE must be greater than zero."
        )

    return (
        (baseline_mae - model_mae)
        / baseline_mae
        * 100
    )