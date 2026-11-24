
"""
STRATUM - Linear Regression Models

Provides reusable Linear Regression and Ridge
Regression pipelines for semiconductor virtual
metrology.

Supports:
- APC-only inputs
- APC and equipment inputs
- Training-only preprocessing
- Numeric feature standardisation
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

from sklearn.pipeline import Pipeline

from ml.preprocessing import build_preprocessor


def build_linear_model(
    numeric_features,
    model_type="linear",
    include_equipment=True,
    alpha=1.0,
):
    """
    Build an unfitted regression pipeline.

    Parameters
    ----------
    numeric_features:
        List of numeric APC feature names.

    model_type:
        "linear" or "ridge".

    include_equipment:
        Whether equipment metadata is included.

    alpha:
        Ridge regularisation strength.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        include_equipment=include_equipment,
        scale_numeric=True,
    )

    if model_type == "linear":

        estimator = LinearRegression()

    elif model_type == "ridge":

        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(
                "Ridge alpha must be finite and positive."
            )

        estimator = Ridge(
            alpha=alpha,
            solver="svd",
        )

    else:

        raise ValueError(
            f"Unsupported model type: {model_type}"
        )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )

    return pipeline


def extract_linear_coefficients(fitted_pipeline):
    """
    Extract transformed feature names and corresponding
    coefficients from a fitted linear model.

    Coefficients represent the transformed feature scale,
    not necessarily the original physical sensor scale.
    """

    preprocessor = (
        fitted_pipeline.named_steps["preprocessor"]
    )

    estimator = (
        fitted_pipeline.named_steps["regressor"]
    )

    feature_names = (
        preprocessor.get_feature_names_out()
    )

    coefficients = np.asarray(
        estimator.coef_
    ).ravel()

    if len(feature_names) != len(coefficients):
        raise ValueError(
            "Feature and coefficient counts do not match."
        )

    coefficient_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
    })

    coefficient_df["Absolute Coefficient"] = (
        coefficient_df["Coefficient"].abs()
    )

    return (
        coefficient_df
        .sort_values(
            "Absolute Coefficient",
            ascending=False,
        )
        .reset_index(drop=True)
    )