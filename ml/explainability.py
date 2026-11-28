
"""
STRATUM - Model Explainability

Reusable explainability utilities for the
thickness and sheet-resistance virtual
metrology models.

Supports:
- XGBoost gain-based importance.
- Native XGBoost TreeSHAP contributions.
- Original-scale resistance explanations.
- Raw-input contribution aggregation.
- Global and individual prediction explanations.
- Numerical additivity verification.

Feature contributions describe model behaviour
and do not establish physical causality.
"""

import numpy as np
import pandas as pd

import xgboost as xgb

from sklearn.pipeline import Pipeline

from sklearn.compose import (
    TransformedTargetRegressor,
)

from sklearn.preprocessing import StandardScaler


def unwrap_fitted_pipeline(model):

    """
    Extract the fitted regression pipeline
    and target transformation parameters.
    """

    if isinstance(
        model,
        TransformedTargetRegressor,
    ):

        if not isinstance(
            model.transformer_,
            StandardScaler,
        ):

            raise TypeError(
                "Only StandardScaler target "
                "transformations are supported."
            )

        pipeline = model.regressor_

        target_mean = float(
            model.transformer_.mean_[0]
        )

        target_scale = float(
            model.transformer_.scale_[0]
        )

    else:

        pipeline = model

        target_mean = 0.0

        target_scale = 1.0

    if not isinstance(pipeline, Pipeline):

        raise TypeError(
            "Expected a fitted sklearn Pipeline."
        )

    required_steps = {
        "preprocessor",
        "regressor",
    }

    if not required_steps.issubset(
        pipeline.named_steps
    ):

        raise ValueError(
            "Required pipeline steps are missing."
        )

    return (
        pipeline,
        target_mean,
        target_scale,
    )


def get_transformed_feature_names(model):

    """
    Retrieve fitted preprocessing output names.
    """

    pipeline, _, _ = (
        unwrap_fitted_pipeline(model)
    )

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    names = [
        str(name)

        for name
        in preprocessor.get_feature_names_out()
    ]

    if len(names) != len(set(names)):

        raise ValueError(
            "Transformed feature names are not unique."
        )

    return names


def extract_gain_importance(model):

    """
    Extract normalised average-gain importance
    from a fitted XGBoost regressor.

    Zero-importance features are retained.
    """

    pipeline, _, _ = (
        unwrap_fitted_pipeline(model)
    )

    regressor = pipeline.named_steps[
        "regressor"
    ]

    booster = regressor.get_booster()

    names = get_transformed_feature_names(
        model
    )

    if booster.num_features() != len(names):

        raise ValueError(
            "Booster feature count does not "
            "match preprocessing output."
        )

    gain_scores = booster.get_score(
        importance_type="gain"
    )

    booster_names = (
        booster.feature_names

        or [
            f"f{i}"

            for i in range(len(names))
        ]
    )

    importance = np.asarray([
        gain_scores.get(
            name,
            0.0,
        )

        for name in booster_names
    ], dtype=float)

    total_gain = importance.sum()

    if total_gain > 0:

        importance = (
            importance / total_gain
        )

    return (
        pd.DataFrame({
            "Feature": names,

            "Gain Importance": importance,
        })
        .sort_values(
            "Gain Importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def calculate_tree_contributions(
    model,
    inputs,
    rtol=1e-5,
    atol=1e-10,
):

    """
    Calculate exact native XGBoost TreeSHAP
    contributions for fitted regression models.

    Return contributions and base predictions
    in the final model's original target scale.

    Verify that the additive explanations
    reconstruct the pipeline predictions.
    """

    if not isinstance(inputs, pd.DataFrame):

        raise TypeError(
            "Inputs must be a pandas DataFrame."
        )

    if inputs.empty:

        raise ValueError(
            "Explainability inputs are empty."
        )

    (
        pipeline,
        target_mean,
        target_scale,
    ) = unwrap_fitted_pipeline(model)

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    regressor = pipeline.named_steps[
        "regressor"
    ]

    transformed_inputs = (
        preprocessor.transform(inputs)
    )

    feature_names = (
        get_transformed_feature_names(model)
    )

    booster = regressor.get_booster()

    if (
        transformed_inputs.shape[1]
        != len(feature_names)
    ):

        raise ValueError(
            "Transformed input dimension mismatch."
        )

    if (
        booster.num_features()
        != len(feature_names)
    ):

        raise ValueError(
            "Booster dimension mismatch."
        )

    matrix = xgb.DMatrix(
        transformed_inputs,

        feature_names=booster.feature_names,
    )

    native_contributions = np.asarray(
        booster.predict(
            matrix,

            pred_contribs=True,

            approx_contribs=False,
        ),

        dtype=float,
    )

    expected_shape = (
        len(inputs),
        len(feature_names) + 1,
    )

    if (
        native_contributions.shape
        != expected_shape
    ):

        raise ValueError(
            "Unexpected contribution matrix shape."
        )

    if not np.isfinite(
        native_contributions
    ).all():

        raise ValueError(
            "Non-finite contributions detected."
        )

    # Convert feature contributions back to
    # the model's original output scale.

    contributions = (
        native_contributions[:, :-1]
        * target_scale
    )

    base_predictions = (
        native_contributions[:, -1]
        * target_scale
        + target_mean
    )

    reconstructed_predictions = (
        base_predictions
        + contributions.sum(axis=1)
    )

    model_predictions = np.asarray(
        model.predict(inputs),

        dtype=float,
    ).ravel()

    np.testing.assert_allclose(
        reconstructed_predictions,

        model_predictions,

        rtol=rtol,

        atol=atol,
    )

    reconstruction_errors = np.abs(
        reconstructed_predictions
        - model_predictions
    )

    return {
        "feature_names": feature_names,

        "contributions": contributions,

        "base_predictions": (
            base_predictions
        ),

        "predictions": model_predictions,

        "reconstructed_predictions": (
            reconstructed_predictions
        ),

        "maximum_reconstruction_error": float(
            reconstruction_errors.max()
        ),

        "mean_reconstruction_error": float(
            reconstruction_errors.mean()
        ),

        "target_mean": target_mean,

        "target_scale": target_scale,
    }


def map_transformed_feature_to_raw(
    transformed_name,
    apc_features,
):

    """
    Map known STRATUM preprocessing outputs
    to their originating raw input.

    Unknown feature names raise an error
    rather than being silently misclassified.
    """

    name = str(transformed_name)

    if "__" in name:

        prefix, name = name.split(
            "__",
            1,
        )

        if prefix == "equipment":

            return "equipment"

    if name.startswith(
        "missingindicator_"
    ):

        name = name.removeprefix(
            "missingindicator_"
        )

    if name in apc_features:

        return name

    if name == "equipment":

        return "equipment"

    if name.startswith(
        "equipment_"
    ):

        return "equipment"

    raise ValueError(
        f"Unrecognised transformed feature: "
        f"{transformed_name}"
    )


def aggregate_raw_contributions(
    explanation,
    apc_features,
):

    """
    Sum signed transformed contributions
    belonging to the same raw input.

    Sensor values and their missingness
    indicators are grouped together.

    Equipment one-hot contributions are
    grouped into the equipment input.

    This is post-hoc aggregation of model
    attributions, not recomputed grouped
    Shapley values.
    """

    transformed_names = (
        explanation["feature_names"]
    )

    contributions = (
        explanation["contributions"]
    )

    raw_names = [
        map_transformed_feature_to_raw(
            name,

            apc_features,
        )

        for name in transformed_names
    ]

    unique_raw_names = list(
        dict.fromkeys(raw_names)
    )

    grouped = np.zeros(
        (
            contributions.shape[0],

            len(unique_raw_names),
        ),

        dtype=float,
    )

    raw_index = {
        name: index

        for index, name

        in enumerate(unique_raw_names)
    }

    for index, raw_name in enumerate(
        raw_names
    ):

        grouped[
            :,

            raw_index[raw_name]
        ] += contributions[:, index]

    np.testing.assert_allclose(
        grouped.sum(axis=1),

        contributions.sum(axis=1),

        rtol=1e-9,

        atol=1e-12,
    )

    return pd.DataFrame(
        grouped,

        columns=unique_raw_names,
    )


def global_shap_importance(
    contribution_frame,
):

    """
    Calculate mean absolute and mean signed
    contributions for every feature.
    """

    importance = pd.DataFrame({
        "Feature": (
            contribution_frame.columns
        ),

        "Mean Absolute Contribution": (
            contribution_frame
            .abs()
            .mean(axis=0)
            .to_numpy()
        ),

        "Mean Signed Contribution": (
            contribution_frame
            .mean(axis=0)
            .to_numpy()
        ),
    })

    total_importance = (
        importance[
            "Mean Absolute Contribution"
        ].sum()
    )

    importance[
        "Relative Importance (%)"
    ] = 0.0

    if total_importance > 0:

        importance[
            "Relative Importance (%)"
        ] = (
            importance[
                "Mean Absolute Contribution"
            ]

            / total_importance

            * 100
        )

    return (
        importance
        .sort_values(
            "Mean Absolute Contribution",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def _json_input_value(value):

    if pd.isna(value):

        return None

    if isinstance(value, np.generic):

        value = value.item()

    if isinstance(value, float):

        if not np.isfinite(value):

            raise ValueError(
                "Non-finite input value."
            )

    return value


def explain_single_prediction(
    model,
    inputs,
    apc_features,
    target,
    top_n=10,
):

    """
    Generate a JSON-compatible local explanation
    for one process observation.

    Include top signed contributions and a
    remainder term for omitted features.

    Actual measured targets are not required.
    """

    if len(inputs) != 1:

        raise ValueError(
            "Expected exactly one input record."
        )

    if top_n < 1:

        raise ValueError(
            "top_n must be positive."
        )

    explanation = (
        calculate_tree_contributions(
            model,

            inputs,
        )
    )

    grouped = aggregate_raw_contributions(
        explanation,

        apc_features,
    )

    contribution_series = grouped.iloc[0]

    ordered_features = (
        contribution_series
        .abs()
        .sort_values(ascending=False)
        .index
    )

    selected_features = (
        ordered_features[:top_n]
    )

    top_features = []

    for name in selected_features:

        value = float(
            contribution_series[name]
        )

        input_value = inputs.iloc[0][name]

        top_features.append({
            "feature": name,

            "input_value": (
                _json_input_value(input_value)
            ),

            "input_missing": bool(
                pd.isna(input_value)
            ),

            "contribution": value,

            "absolute_contribution": abs(value),

            "direction": (
                "increases_prediction"

                if value > 0

                else "decreases_prediction"

                if value < 0

                else "neutral"
            ),
        })

    total_contribution = float(
        contribution_series.sum()
    )

    selected_contribution = sum(
        item["contribution"]

        for item in top_features
    )

    other_contribution = (
        total_contribution
        - selected_contribution
    )

    base_prediction = float(
        explanation[
            "base_predictions"
        ][0]
    )

    prediction = float(
        explanation[
            "predictions"
        ][0]
    )

    reconstructed = (
        base_prediction
        + selected_contribution
        + other_contribution
    )

    return {
        "target": target,

        "method": (
            "XGBoost native TreeSHAP"
        ),

        "output_scale": (
            "Original dataset target scale"
        ),

        "aggregation": (
            "Sum of transformed-feature "
            "attributions by originating raw input"
        ),

        "base_prediction": base_prediction,

        "prediction": prediction,

        "total_feature_contribution": (
            total_contribution
        ),

        "top_features": top_features,

        "other_feature_contribution": (
            other_contribution
        ),

        "reconstructed_prediction": (
            reconstructed
        ),

        "reconstruction_error": abs(
            prediction - reconstructed
        ),

        "interpretation_warning": (
            "Feature contributions explain "
            "model behaviour, not physical causality."
        ),
    }