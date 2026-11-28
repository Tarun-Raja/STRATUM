
"""
STRATUM - Final Held-out Evaluation

Reusable utilities for regression diagnostics,
lot-aware bootstrap uncertainty, paired model
comparisons and input distribution monitoring.

Confidence intervals describe aggregate metric
uncertainty under cluster-resampling assumptions.

They are not individual prediction intervals.
"""

import numpy as np
import pandas as pd

from ml.evaluation import evaluate_regression


def validate_prediction_arrays(y_true, y_pred):

    """
    Verify aligned finite one-dimensional arrays.
    """

    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:

        raise ValueError(
            "Actual and predicted arrays must be 1D."
        )

    if actual.shape != predicted.shape:

        raise ValueError(
            "Actual and predicted shapes differ."
        )

    if len(actual) < 2:

        raise ValueError(
            "At least two observations are required."
        )

    if not np.isfinite(actual).all():

        raise ValueError("Non-finite actual values.")

    if not np.isfinite(predicted).all():

        raise ValueError("Non-finite predictions.")

    return actual, predicted


def regression_diagnostics(y_true, y_pred):

    """
    Return regression metrics and tail errors.
    """

    actual, predicted = validate_prediction_arrays(
        y_true,
        y_pred,
    )

    errors = np.abs(actual - predicted)

    metrics = evaluate_regression(
        actual,
        predicted,
    )

    return {
        **metrics,

        "Records": len(actual),

        "P50 Absolute Error": float(
            np.quantile(errors, 0.50)
        ),

        "P90 Absolute Error": float(
            np.quantile(errors, 0.90)
        ),

        "P95 Absolute Error": float(
            np.quantile(errors, 0.95)
        ),

        "P99 Absolute Error": float(
            np.quantile(errors, 0.99)
        ),

        "Maximum Absolute Error": float(
            errors.max()
        ),

        "Non-Positive Predictions": int(
            (predicted <= 0).sum()
        ),
    }


def build_cluster_indices(groups):

    """
    Construct row-index arrays for each cluster.
    """

    group_series = pd.Series(
        groups
    ).reset_index(drop=True)

    if group_series.isna().any():

        raise ValueError(
            "Bootstrap cluster identifiers cannot be missing."
        )

    cluster_codes, cluster_names = pd.factorize(
        group_series,
        sort=False,
    )

    if len(cluster_names) < 2:

        raise ValueError(
            "At least two independent resampling "
            "clusters are required."
        )

    clusters = [
        np.flatnonzero(cluster_codes == index)

        for index in range(len(cluster_names))
    ]

    return clusters


def generate_cluster_bootstrap_indices(
    groups,
    n_bootstrap=2000,
    random_state=42,
):

    """
    Yield bootstrap row indices by sampling
    complete clusters with replacement.
    """

    if n_bootstrap < 1:

        raise ValueError(
            "n_bootstrap must be positive."
        )

    clusters = build_cluster_indices(groups)

    rng = np.random.default_rng(random_state)

    cluster_count = len(clusters)

    for _ in range(n_bootstrap):

        sampled_clusters = rng.integers(
            low=0,
            high=cluster_count,
            size=cluster_count,
        )

        yield np.concatenate([
            clusters[index]

            for index in sampled_clusters
        ])


def _bootstrap_metrics(actual, predicted):

    """
    Calculate bootstrap statistics without
    repeatedly invoking sklearn metric wrappers.
    """

    residual = predicted - actual

    absolute_error = np.abs(residual)

    squared_error = residual ** 2

    denominator = np.sum(
        (actual - actual.mean()) ** 2
    )

    r2 = (
        1.0 - squared_error.sum() / denominator

        if denominator > 0

        else np.nan
    )

    mape = (
        np.mean(
            absolute_error / np.abs(actual)
        ) * 100

        if np.all(actual != 0)

        else np.nan
    )

    return {
        "MAE": float(absolute_error.mean()),

        "RMSE": float(
            np.sqrt(squared_error.mean())
        ),

        "R2": float(r2),

        "MAPE (%)": float(mape),

        "Bias": float(residual.mean()),
    }


def _summarise_bootstrap_values(
    values,
    confidence_level,
):

    values = np.asarray(values, dtype=float)

    finite = values[np.isfinite(values)]

    if len(finite) < 2:

        raise ValueError(
            "Insufficient finite bootstrap statistics."
        )

    alpha = 1.0 - confidence_level

    return {
        "Lower CI": float(
            np.quantile(finite, alpha / 2)
        ),

        "Upper CI": float(
            np.quantile(finite, 1 - alpha / 2)
        ),

        "Bootstrap Mean": float(finite.mean()),

        "Bootstrap Std": float(
            finite.std(ddof=1)
        ),

        "Valid Replicates": len(finite),
    }


def cluster_bootstrap_regression(
    y_true,
    y_pred,
    groups,
    n_bootstrap=2000,
    confidence_level=0.95,
    random_state=42,
):

    """
    Estimate percentile confidence intervals
    for fixed-model aggregate regression metrics.
    """

    actual, predicted = validate_prediction_arrays(
        y_true,
        y_pred,
    )

    if len(groups) != len(actual):

        raise ValueError(
            "Cluster identifiers must align with predictions."
        )

    if not 0 < confidence_level < 1:

        raise ValueError(
            "confidence_level must be between zero and one."
        )

    point_estimates = _bootstrap_metrics(
        actual,
        predicted,
    )

    bootstrap_values = {
        metric: []

        for metric in point_estimates
    }

    for indices in generate_cluster_bootstrap_indices(
        groups,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    ):

        statistics = _bootstrap_metrics(
            actual[indices],
            predicted[indices],
        )

        for metric, value in statistics.items():

            bootstrap_values[metric].append(value)

    cluster_count = len(
        build_cluster_indices(groups)
    )

    rows = []

    for metric, point in point_estimates.items():

        interval = _summarise_bootstrap_values(
            bootstrap_values[metric],
            confidence_level,
        )

        rows.append({
            "Metric": metric,

            "Point Estimate": point,

            **interval,

            "Confidence Level": confidence_level,

            "Bootstrap Repetitions": n_bootstrap,

            "Clusters": cluster_count,
        })

    return pd.DataFrame(rows)


def paired_cluster_bootstrap(
    y_true,
    reference_predictions,
    candidate_predictions,
    groups,
    n_bootstrap=2000,
    confidence_level=0.95,
    random_state=42,
):

    """
    Compare paired absolute prediction errors.

    Positive MAE reduction favours the candidate.

    The same resampled observations are used
    for the reference and candidate in each
    bootstrap repetition.
    """

    actual, reference = validate_prediction_arrays(
        y_true,
        reference_predictions,
    )

    _, candidate = validate_prediction_arrays(
        y_true,
        candidate_predictions,
    )

    if len(groups) != len(actual):

        raise ValueError(
            "Cluster identifiers must align with predictions."
        )

    if not 0 < confidence_level < 1:

        raise ValueError(
            "Invalid confidence level."
        )

    reference_errors = np.abs(
        actual - reference
    )

    candidate_errors = np.abs(
        actual - candidate
    )

    differences = (
        reference_errors - candidate_errors
    )

    reference_mae = float(
        reference_errors.mean()
    )

    candidate_mae = float(
        candidate_errors.mean()
    )

    if reference_mae <= 0:

        raise ValueError(
            "Reference MAE must be positive."
        )

    bootstrap_differences = []

    bootstrap_improvements = []

    for indices in generate_cluster_bootstrap_indices(
        groups,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    ):

        sampled_reference = float(
            reference_errors[indices].mean()
        )

        sampled_candidate = float(
            candidate_errors[indices].mean()
        )

        difference = (
            sampled_reference - sampled_candidate
        )

        bootstrap_differences.append(
            difference
        )

        if sampled_reference > 0:

            bootstrap_improvements.append(
                difference / sampled_reference * 100
            )

    difference_interval = (
        _summarise_bootstrap_values(
            bootstrap_differences,
            confidence_level,
        )
    )

    improvement_interval = (
        _summarise_bootstrap_values(
            bootstrap_improvements,
            confidence_level,
        )
    )

    wins = candidate_errors < reference_errors

    losses = candidate_errors > reference_errors

    ties = candidate_errors == reference_errors

    return {
        "Records": len(actual),

        "Clusters": len(
            build_cluster_indices(groups)
        ),

        "Reference MAE": reference_mae,

        "Candidate MAE": candidate_mae,

        "MAE Reduction": (
            reference_mae - candidate_mae
        ),

        "MAE Reduction Lower CI": (
            difference_interval["Lower CI"]
        ),

        "MAE Reduction Upper CI": (
            difference_interval["Upper CI"]
        ),

        "MAE Improvement (%)": (
            (
                reference_mae - candidate_mae
            )
            / reference_mae
            * 100
        ),

        "Improvement Lower CI (%)": (
            improvement_interval["Lower CI"]
        ),

        "Improvement Upper CI (%)": (
            improvement_interval["Upper CI"]
        ),

        "Candidate Wins": int(wins.sum()),

        "Candidate Losses": int(losses.sum()),

        "Ties": int(ties.sum()),

        "Win Rate (%)": float(
            wins.mean() * 100
        ),

        "Confidence Level": confidence_level,

        "Bootstrap Repetitions": n_bootstrap,
    }


def evaluate_subgroups(
    y_true,
    y_pred,
    groups,
):

    """
    Evaluate fixed-model prediction behaviour
    across observed subgroups.
    """

    actual, predicted = validate_prediction_arrays(
        y_true,
        y_pred,
    )

    if len(groups) != len(actual):

        raise ValueError(
            "Subgroup labels must align with observations."
        )

    frame = pd.DataFrame({
        "Group": np.asarray(groups),

        "Actual": actual,

        "Predicted": predicted,
    })

    rows = []

    for label, group in frame.groupby(
        "Group",
        dropna=False,
        sort=False,
    ):

        residuals = (
            group["Predicted"]
            - group["Actual"]
        ).to_numpy()

        absolute_errors = np.abs(residuals)

        denominator = np.sum(
            (
                group["Actual"]
                - group["Actual"].mean()
            ) ** 2
        )

        group_r2 = (
            1 - np.sum(residuals ** 2) / denominator

            if len(group) >= 2 and denominator > 0

            else np.nan
        )

        rows.append({
            "Group": str(label),

            "Records": len(group),

            "Actual Mean": float(
                group["Actual"].mean()
            ),

            "Predicted Mean": float(
                group["Predicted"].mean()
            ),

            "MAE": float(
                absolute_errors.mean()
            ),

            "RMSE": float(
                np.sqrt(
                    np.mean(residuals ** 2)
                )
            ),

            "R2": float(group_r2),

            "Bias": float(residuals.mean()),

            "P95 Absolute Error": float(
                np.quantile(absolute_errors, 0.95)
            ),

            "Maximum Absolute Error": float(
                absolute_errors.max()
            ),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(
            "MAE",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def fit_input_warning_profile(
    training_inputs,
    apc_features,
    z_threshold=5.0,
):

    """
    Fit descriptive sensor statistics using
    eligible training observations only.

    Missing inputs are not interpreted as
    observed extreme sensor measurements.

    These warnings are heuristic and do not
    represent calibrated prediction confidence.
    """

    if z_threshold <= 0:

        raise ValueError(
            "z_threshold must be positive."
        )

    frame = (
        training_inputs[apc_features]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )

    means = frame.mean()

    scales = frame.std(ddof=0)

    missing_counts = frame.isna().sum(axis=1)

    feature_statistics = {}

    for feature in apc_features:

        mean = means[feature]

        scale = scales[feature]

        feature_statistics[feature] = {
            "mean": (
                float(mean)

                if np.isfinite(mean)

                else None
            ),

            "std": (
                float(scale)

                if np.isfinite(scale)

                else None
            ),

            "observed_records": int(
                frame[feature].notna().sum()
            ),

            "missing_fraction": float(
                frame[feature].isna().mean()
            ),
        }

    known_equipment = []

    if "equipment" in training_inputs.columns:

        known_equipment = sorted(
            training_inputs["equipment"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

    return {
        "z_threshold": float(z_threshold),

        "feature_statistics": feature_statistics,

        "training_missing_count_p95": float(
            missing_counts.quantile(0.95)
        ),

        "training_missing_count_maximum": int(
            missing_counts.max()
        ),

        "known_equipment": known_equipment,

        "warning_type": (
            "Training-distribution heuristic; "
            "not calibrated uncertainty"
        ),
    }


def calculate_input_warnings(
    inputs,
    profile,
):

    """
    Identify observed extreme sensor values,
    elevated missingness and unknown equipment.
    """

    features = list(
        profile["feature_statistics"]
    )

    frame = (
        inputs[features]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )

    means = pd.Series({
        feature: stats["mean"]

        for feature, stats in
        profile["feature_statistics"].items()
    }, dtype=float)

    scales = pd.Series({
        feature: stats["std"]

        for feature, stats in
        profile["feature_statistics"].items()
    }, dtype=float)

    valid_scales = scales.where(
        np.isfinite(scales) & (scales > 0)
    )

    z_scores = (
        (frame - means)
        / valid_scales
    ).abs()

    extreme = (
        z_scores > profile["z_threshold"]
    )

    zero_scale = scales.eq(0)

    constant_deviation = (
        frame.notna()
        & frame.ne(means)
        & zero_scale
    )

    unsupported_features = means.isna()

    unsupported_observed = (
        frame.notna()
        & unsupported_features
    )

    missing_count = frame.isna().sum(axis=1)

    result = pd.DataFrame({
        "Missing APC Inputs": missing_count,

        "Extreme APC Inputs": extreme.sum(axis=1),

        "Constant Feature Deviations": (
            constant_deviation.sum(axis=1)
        ),

        "Previously Unobserved Features": (
            unsupported_observed.sum(axis=1)
        ),
    }, index=inputs.index)

    result["Elevated Missingness"] = (
        missing_count
        > profile["training_missing_count_p95"]
    )

    result["Extreme Sensor Warning"] = (
        extreme.any(axis=1)
        | constant_deviation.any(axis=1)
        | unsupported_observed.any(axis=1)
    )

    if "equipment" in inputs.columns:

        equipment = inputs["equipment"]

        result["Unknown Equipment"] = (
            equipment.notna()

            & ~equipment.astype(str).isin(
                profile["known_equipment"]
            )
        )

        result["Missing Equipment"] = (
            equipment.isna()
        )

    else:

        result["Unknown Equipment"] = False

        result["Missing Equipment"] = True

    result["Any Input Warning"] = (
        result["Elevated Missingness"]
        | result["Extreme Sensor Warning"]
        | result["Unknown Equipment"]
        | result["Missing Equipment"]
    )

    return result