
"""
STRATUM - Random Forest Regression

Reusable nonlinear virtual metrology regression
pipelines for semiconductor manufacturing data.

Supports:
- APC-only input configuration.
- APC and equipment input configuration.
- Training-only median imputation.
- Numeric missingness indicators.
- Optional equipment encoding.
- Reproducible Random Forest experiments.
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from ml.preprocessing import build_preprocessor


def build_random_forest_model(
    numeric_features,
    include_equipment=True,
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    max_features=0.7,
    random_state=42,
    n_jobs=-1,
):
    """
    Construct an unfitted Random Forest pipeline.

    Parameters
    ----------
    numeric_features:
        Numeric APC input column names.

    include_equipment:
        Whether to include equipment metadata.

    n_estimators:
        Number of trees in the forest.

    max_depth:
        Maximum depth of individual trees.
        None allows unrestricted depth.

    min_samples_leaf:
        Minimum number of training samples
        required in each leaf.

    max_features:
        Fraction of features considered at each split.

    random_state:
        Random seed for reproducibility.

    n_jobs:
        Number of CPU workers.
        -1 uses all available logical CPU cores.
    """

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        include_equipment=include_equipment,
        scale_numeric=False,
        clip_z=None,
    )

    estimator = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        criterion="squared_error",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", estimator),
        ]
    )

    return pipeline


def extract_forest_feature_importance(
    fitted_pipeline,
):
    """
    Extract impurity-based feature importance
    from a fitted Random Forest pipeline.

    Importance values refer to transformed features.

    They do not represent causal manufacturing effects
    and may favour certain types of predictors.
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