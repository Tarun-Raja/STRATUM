
"""
STRATUM - XGBoost Regression

Reusable gradient-boosted regression pipelines
for semiconductor virtual metrology.

Supports:
- APC-only inputs.
- APC and equipment inputs.
- Median-imputed numeric inputs.
- Native XGBoost missing-value handling.
- Reproducible CPU-based training.
"""

import numpy as np
import pandas as pd

from xgboost import XGBRegressor

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ml.preprocessing import (
    build_preprocessor,
    CATEGORICAL_FEATURES,
)


def build_native_missing_preprocessor(
    numeric_features,
    include_equipment=True,
):
    """
    Preserve missing numerical APC measurements.

    Equipment metadata is optionally imputed
    and one-hot encoded.

    Numeric NaN values are passed directly
    to XGBoost.
    """

    transformers = [
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]

    if include_equipment:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "equipment",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            )
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )


def build_xgboost_model(
    numeric_features,
    include_equipment=True,
    missing_strategy="median",
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    gamma=0.0,
    random_state=42,
    n_jobs=-1,
):
    """
    Construct an unfitted XGBoost regression pipeline.

    missing_strategy:
        "median" or "native".

    All learned preprocessing transformations
    must be fitted using training data only.
    """

    if missing_strategy == "median":

        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            include_equipment=include_equipment,
            scale_numeric=False,
            clip_z=None,
        )

    elif missing_strategy == "native":

        preprocessor = (
            build_native_missing_preprocessor(
                numeric_features=numeric_features,
                include_equipment=include_equipment,
            )
        )

    else:

        raise ValueError(
            f"Unsupported missing strategy: "
            f"{missing_strategy}"
        )

    estimator = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        device="cpu",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        gamma=gamma,
        random_state=random_state,
        n_jobs=n_jobs,
        missing=np.nan,
        importance_type="gain",
        verbosity=0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )


def extract_xgboost_feature_importance(
    fitted_pipeline,
):
    """
    Extract normalised gain-based importance.

    Importance values describe the fitted model,
    not causal manufacturing relationships.
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

    importance_values = np.asarray(
        estimator.feature_importances_
    )

    if len(feature_names) != len(importance_values):

        raise ValueError(
            "Feature and importance counts do not match."
        )

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance_values,
    })

    return (
        importance_df
        .sort_values(
            "Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )