
"""
STRATUM - Chronological Cross-validation

Reusable model evaluation utilities for
semiconductor virtual metrology.

Features:
- Expanding-window chronological folds.
- Lot and batch overlap purging.
- Training-only preprocessing.
- Internal XGBoost early stopping.
- Reproducible regression candidate construction.
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit

from ml.baselines import EquipmentMeanRegressor
from ml.linear_models import build_linear_model
from ml.random_forest_models import (
    build_random_forest_model,
)
from ml.xgboost_models import build_xgboost_model


GROUP_COLUMNS = ("lot", "bat_id")


def prepare_chronological_data(data):

    required_columns = [
        "wafer_id",
        "runstart",
        *GROUP_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Missing required columns: "
            f"{missing_columns}"
        )

    ordered = data.copy()

    ordered["runstart"] = pd.to_datetime(
        ordered["runstart"],
        errors="raise",
    )

    if ordered[required_columns].isna().any().any():

        raise ValueError(
            "Missing chronological or group identifiers."
        )

    if not ordered["wafer_id"].is_unique:

        raise ValueError(
            "Wafer identifiers must be unique."
        )

    return (
        ordered
        .sort_values(["runstart", "wafer_id"])
        .reset_index(drop=True)
    )


def purge_training_overlap(
    training_data,
    validation_data,
    group_columns=GROUP_COLUMNS,
):

    if training_data.empty or validation_data.empty:

        raise ValueError(
            "Training and validation must be nonempty."
        )

    # Enforce strict chronological separation.

    validation_start = (
        validation_data["runstart"].min()
    )

    time_overlap = (
        training_data["runstart"]
        >= validation_start
    )

    # Identify overlapping manufacturing groups.

    group_masks = {}

    for column in group_columns:

        validation_groups = set(
            validation_data[column]
            .dropna()
            .unique()
        )

        group_masks[column] = (
            training_data[column]
            .isin(validation_groups)
        )

    remove_mask = time_overlap.copy()

    for mask in group_masks.values():

        remove_mask = remove_mask | mask

    purged_training = (
        training_data
        .loc[~remove_mask]
        .copy()
    )

    if purged_training.empty:

        raise ValueError(
            "No training records remain after purging."
        )

    assert (
        purged_training["runstart"].max()
        < validation_data["runstart"].min()
    )

    for column in group_columns:

        assert set(
            purged_training[column]
        ).isdisjoint(
            set(validation_data[column])
        )

    assert set(
        purged_training["wafer_id"]
    ).isdisjoint(
        set(validation_data["wafer_id"])
    )

    purge_info = {
        "initial_training_records": len(
            training_data
        ),

        "training_records": len(
            purged_training
        ),

        "purged_records": int(
            remove_mask.sum()
        ),

        "timestamp_overlap_records": int(
            time_overlap.sum()
        ),

        "group_overlap_records": {
            column: int(mask.sum())

            for column, mask
            in group_masks.items()
        },
    }

    return purged_training, purge_info


def chronological_cv_folds(
    data,
    n_splits=5,
    initial_train_fraction=0.4,
):

    if not 0 < initial_train_fraction < 1:

        raise ValueError(
            "Initial training fraction must be "
            "between zero and one."
        )

    if n_splits < 2:

        raise ValueError(
            "At least two folds are required."
        )

    ordered = prepare_chronological_data(
        data
    )

    validation_size = int(
        len(ordered)
        * (1 - initial_train_fraction)
        / n_splits
    )

    if validation_size < 2:

        raise ValueError(
            "Insufficient records per validation fold."
        )

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        test_size=validation_size,
    )

    folds = []

    for fold_number, (
        train_indices,
        validation_indices,
    ) in enumerate(
        splitter.split(ordered),
        start=1,
    ):

        initial_training = ordered.iloc[
            train_indices
        ].copy()

        validation = ordered.iloc[
            validation_indices
        ].copy()

        training, purge_info = (
            purge_training_overlap(
                initial_training,
                validation,
            )
        )

        fold_info = {
            "fold": fold_number,

            **purge_info,

            "validation_records": len(
                validation
            ),

            "training_lots": int(
                training["lot"].nunique()
            ),

            "validation_lots": int(
                validation["lot"].nunique()
            ),

            "training_start": str(
                training["runstart"].min()
            ),

            "training_end": str(
                training["runstart"].max()
            ),

            "validation_start": str(
                validation["runstart"].min()
            ),

            "validation_end": str(
                validation["runstart"].max()
            ),
        }

        folds.append({
            "fold": fold_number,
            "train": training,
            "validation": validation,
            "info": fold_info,
        })

    return folds


def chronological_inner_split(
    data,
    validation_fraction=0.2,
):

    if not 0 < validation_fraction < 1:

        raise ValueError(
            "Invalid inner validation fraction."
        )

    ordered = prepare_chronological_data(
        data
    )

    boundary = int(
        len(ordered)
        * (1 - validation_fraction)
    )

    initial_training = ordered.iloc[
        :boundary
    ].copy()

    validation = ordered.iloc[
        boundary:
    ].copy()

    training, purge_info = (
        purge_training_overlap(
            initial_training,
            validation,
        )
    )

    return training, validation, purge_info


def get_candidate_columns(
    candidate,
    apc_features,
):

    if candidate["algorithm"] == "Equipment Mean":

        return ["equipment"]

    columns = list(apc_features)

    if candidate["include_equipment"]:

        columns.append("equipment")

    return columns


def build_candidate_model(
    candidate,
    apc_features,
    n_jobs=-1,
    boosting_rounds=None,
):

    algorithm = candidate["algorithm"]

    parameters = candidate[
        "parameters"
    ].copy()

    include_equipment = candidate[
        "include_equipment"
    ]

    if algorithm == "Equipment Mean":

        return EquipmentMeanRegressor()

    if algorithm == "Ridge":

        return build_linear_model(
            numeric_features=apc_features,
            model_type="ridge",
            include_equipment=include_equipment,
            **parameters,
        )

    if algorithm == "Random Forest":

        return build_random_forest_model(
            numeric_features=apc_features,
            include_equipment=include_equipment,
            n_jobs=n_jobs,
            **parameters,
        )

    if algorithm == "XGBoost":

        if boosting_rounds is not None:

            parameters["n_estimators"] = int(
                boosting_rounds
            )

        return build_xgboost_model(
            numeric_features=apc_features,
            include_equipment=include_equipment,
            n_jobs=n_jobs,
            **parameters,
        )

    raise ValueError(
        f"Unsupported algorithm: {algorithm}"
    )


def fit_candidate_model(
    candidate,
    training_data,
    apc_features,
    target,
    n_jobs=-1,
    early_stopping_rounds=50,
    inner_validation_fraction=0.2,
    fixed_boosting_rounds=None,
):

    """
    Fit a candidate using training observations only.

    XGBoost can select boosting rounds using a
    chronological inner validation subset.

    After round selection, a fresh pipeline is
    refitted on all eligible outer training records.
    """

    columns = get_candidate_columns(
        candidate,
        apc_features,
    )

    fit_info = {
        "selected_boosting_rounds": None,
        "inner_training_records": None,
        "inner_validation_records": None,
        "inner_purged_records": None,
        "boosting_budget_reached": None,
        "inner_best_mae": None,
    }

    selected_rounds = fixed_boosting_rounds

    if (
        candidate["algorithm"] == "XGBoost"
        and selected_rounds is None
    ):

        inner_train, inner_val, purge_info = (
            chronological_inner_split(
                training_data,
                validation_fraction=(
                    inner_validation_fraction
                ),
            )
        )

        stopping_pipeline = build_candidate_model(
            candidate=candidate,
            apc_features=apc_features,
            n_jobs=n_jobs,
        )

        preprocessor = (
            stopping_pipeline.named_steps[
                "preprocessor"
            ]
        )

        regressor = (
            stopping_pipeline.named_steps[
                "regressor"
            ]
        )

        X_inner_train = (
            preprocessor.fit_transform(
                inner_train[columns]
            )
        )

        X_inner_val = preprocessor.transform(
            inner_val[columns]
        )

        regressor.set_params(
            early_stopping_rounds=(
                early_stopping_rounds
            ),
            eval_metric="mae",
        )

        regressor.fit(
            X_inner_train,
            inner_train[target],

            eval_set=[
                (
                    X_inner_val,
                    inner_val[target],
                )
            ],

            verbose=False,
        )

        selected_rounds = int(
            regressor.best_iteration + 1
        )

        budget = int(
            candidate["parameters"][
                "n_estimators"
            ]
        )

        actual_rounds = (
            regressor.get_booster()
            .num_boosted_rounds()
        )

        fit_info.update({
            "selected_boosting_rounds": (
                selected_rounds
            ),

            "inner_training_records": len(
                inner_train
            ),

            "inner_validation_records": len(
                inner_val
            ),

            "inner_purged_records": (
                purge_info["purged_records"]
            ),

            "boosting_budget_reached": (
                actual_rounds >= budget
            ),

            "inner_best_mae": float(
                regressor.best_score
            ),
        })

    if (
        candidate["algorithm"] == "XGBoost"
        and selected_rounds is not None
    ):

        fit_info[
            "selected_boosting_rounds"
        ] = int(selected_rounds)

    final_pipeline = build_candidate_model(
        candidate=candidate,
        apc_features=apc_features,
        n_jobs=n_jobs,
        boosting_rounds=selected_rounds,
    )

    final_pipeline.fit(
        training_data[columns],
        training_data[target],
    )

    return final_pipeline, fit_info