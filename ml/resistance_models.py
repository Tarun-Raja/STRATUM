
"""
STRATUM - Sheet Resistance Virtual Metrology

Provides resistance-specific baseline models,
training-fitted target standardisation, and
chronological XGBoost early stopping.

All public predictions are returned in the
original dataset target scale.
"""

import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor

from ml.cross_validation import (
    chronological_inner_split,
    get_candidate_columns,
    build_candidate_model,
)


RESISTANCE_TARGET = "mean_resistance"


def get_resistance_candidate_columns(
    candidate,
    apc_features,
):

    if candidate["algorithm"] in (
        "Global Mean",
        "Global Median",
    ):

        # DummyRegressor ignores the input values.
        # A consistent non-target input is provided.

        return [apc_features[0]]

    return get_candidate_columns(
        candidate,
        apc_features,
    )


def fit_resistance_candidate(
    candidate,
    training_data,
    apc_features,
    target=RESISTANCE_TARGET,
    n_jobs=-1,
    early_stopping_rounds=50,
    inner_validation_fraction=0.2,
    fixed_boosting_rounds=None,
):

    """
    Fit a resistance candidate using only
    the supplied training observations.

    Baselines operate in the original target scale.

    ML regressors use a StandardScaler fitted on
    training targets and automatically inverse
    transform their predictions.

    XGBoost early stopping uses a separately fitted
    input preprocessor and target scaler within
    the chronological inner training subset.
    """

    columns = get_resistance_candidate_columns(
        candidate,
        apc_features,
    )

    y_train = training_data[target].to_numpy(
        dtype=float
    )

    if not np.isfinite(y_train).all():

        raise ValueError(
            "Non-finite resistance targets detected."
        )

    if (y_train <= 0).any():

        raise ValueError(
            "Resistance targets must be positive."
        )

    algorithm = candidate["algorithm"]

    fit_info = {
        "target_standardised": False,
        "target_training_mean": float(
            y_train.mean()
        ),
        "target_training_std": float(
            y_train.std(ddof=0)
        ),
        "selected_boosting_rounds": None,
        "inner_training_records": None,
        "inner_validation_records": None,
        "inner_purged_records": None,
        "inner_best_mae_original_scale": None,
        "boosting_budget_reached": None,
    }

    # Historical global baselines

    if algorithm in (
        "Global Mean",
        "Global Median",
    ):

        strategy = (
            "mean"
            if algorithm == "Global Mean"
            else "median"
        )

        model = DummyRegressor(
            strategy=strategy
        )

        model.fit(
            training_data[columns],
            y_train,
        )

        return model, fit_info

    # Equipment-specific historical baseline

    if algorithm == "Equipment Mean":

        model = build_candidate_model(
            candidate=candidate,
            apc_features=apc_features,
            n_jobs=n_jobs,
        )

        model.fit(
            training_data[columns],
            y_train,
        )

        return model, fit_info

    if y_train.std(ddof=0) <= 0:

        raise ValueError(
            "Training target is constant."
        )

    selected_rounds = fixed_boosting_rounds

    fit_info["target_standardised"] = True

    # Internal XGBoost early stopping

    if (
        algorithm == "XGBoost"
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

        # Fit target scaler exclusively on
        # the inner training targets.

        target_scaler = StandardScaler()

        y_inner_train_raw = (
            inner_train[target]
            .to_numpy(dtype=float)
            .reshape(-1, 1)
        )

        y_inner_val_raw = (
            inner_val[target]
            .to_numpy(dtype=float)
            .reshape(-1, 1)
        )

        if y_inner_train_raw.std() <= 0:

            raise ValueError(
                "Inner training target is constant."
            )

        y_inner_train = (
            target_scaler.fit_transform(
                y_inner_train_raw
            ).ravel()
        )

        y_inner_val = (
            target_scaler.transform(
                y_inner_val_raw
            ).ravel()
        )

        regressor.set_params(
            early_stopping_rounds=(
                early_stopping_rounds
            ),
            eval_metric="mae",
        )

        regressor.fit(
            X_inner_train,
            y_inner_train,

            eval_set=[
                (
                    X_inner_val,
                    y_inner_val,
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

        # A standardised MAE is converted back
        # by multiplying by the training scale.

        inner_mae_original_scale = (
            float(regressor.best_score)
            * float(target_scaler.scale_[0])
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

            "inner_best_mae_original_scale": (
                inner_mae_original_scale
            ),

            "boosting_budget_reached": (
                actual_rounds >= budget
            ),
        })

    if algorithm == "XGBoost":

        if selected_rounds is None:

            raise ValueError(
                "Boosting-round selection failed."
            )

        fit_info[
            "selected_boosting_rounds"
        ] = int(selected_rounds)

    # Fresh model using the selected configuration

    regression_pipeline = build_candidate_model(
        candidate=candidate,
        apc_features=apc_features,
        n_jobs=n_jobs,
        boosting_rounds=selected_rounds,
    )

    # Refit the complete pipeline and target scaler
    # on all eligible outer training observations.

    model = TransformedTargetRegressor(
        regressor=regression_pipeline,
        transformer=StandardScaler(),
        check_inverse=True,
    )

    model.fit(
        training_data[columns],
        y_train,
    )

    return model, fit_info


def get_fitted_resistance_pipeline(model):

    """
    Return the fitted input-processing and regression
    pipeline from a trained resistance model.

    Intended for explainability and diagnostics.
    """

    if not isinstance(
        model,
        TransformedTargetRegressor,
    ):

        raise TypeError(
            "Expected a target-standardised "
            "resistance ML model."
        )

    return model.regressor_