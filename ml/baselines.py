
"""
STRATUM - Baseline Regression Models

Provides a simple equipment-specific prediction benchmark.

The model learns average target values for each
equipment identifier using training data only.

For unknown equipment, it falls back to the
global training-set mean.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class EquipmentMeanRegressor(
    RegressorMixin,
    BaseEstimator,
):
    """
    Predict the average training target for each
    equipment identifier.
    """

    def __init__(self, equipment_column="equipment"):
        self.equipment_column = equipment_column

    def fit(self, X, y):
        """
        Learn equipment-specific target averages
        using training observations only.
        """

        if self.equipment_column not in X.columns:
            raise ValueError(
                "Equipment column missing from inputs."
            )

        y_array = np.asarray(y, dtype=float)

        if y_array.ndim != 1:
            raise ValueError(
                "Target must be one-dimensional."
            )

        if len(X) != len(y_array):
            raise ValueError(
                "Input and target lengths do not match."
            )

        if len(y_array) == 0:
            raise ValueError(
                "Cannot fit on an empty dataset."
            )

        if not np.isfinite(y_array).all():
            raise ValueError(
                "Training target contains invalid values."
            )

        training_data = pd.DataFrame({
            "equipment": (
                X[self.equipment_column].to_numpy()
            ),
            "target": y_array,
        })

        # Global fallback prediction
        self.global_mean_ = float(
            training_data["target"].mean()
        )

        # Equipment-specific predictions
        self.equipment_means_ = (
            training_data
            .groupby("equipment")["target"]
            .mean()
            .to_dict()
        )

        return self

    def predict(self, X):
        """
        Predict using training-set equipment averages.

        Unknown or missing equipment identifiers
        use the global training-set mean.
        """

        check_is_fitted(
            self,
            [
                "global_mean_",
                "equipment_means_",
            ],
        )

        if self.equipment_column not in X.columns:
            raise ValueError(
                "Equipment column missing from inputs."
            )

        predictions = (
            X[self.equipment_column]
            .map(self.equipment_means_)
            .fillna(self.global_mean_)
        )

        return predictions.to_numpy(dtype=float)
