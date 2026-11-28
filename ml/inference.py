
"""
STRATUM - Dual-target Inference Engine.

Validate process inputs, generate material-property
predictions, calculate distribution warnings and
produce local TreeSHAP evidence.

No model training occurs during inference.
"""

from numbers import Real

import numpy as np
import pandas as pd

from ml.model_registry import (
    TARGETS,
    load_model_bundle,
)

from ml.final_evaluation import (
    calculate_input_warnings,
)

from ml.explainability import (
    explain_single_prediction,
)


class InputValidationError(ValueError):

    """Invalid process input or input schema."""


class PredictionError(RuntimeError):

    """Numerically invalid model output."""


class StratumInferenceEngine:

    def __init__(self, bundle_path):

        bundle = load_model_bundle(
            bundle_path
        )

        self.models = bundle["models"]

        self.metadata = bundle["metadata"]

        self.input_schema = bundle["input_schema"]

        self.warning_profile = bundle["warning_profile"]

        self.global_explainability = (
            bundle["global_explainability"]
        )

        self.evaluation = bundle["evaluation"]

        self.model_version = (
            self.metadata["model_version"]
        )

        self.apc_features = list(
            self.input_schema["apc_features"]
        )

        self.input_columns = (
            self.input_schema["model_inputs"]
        )

        self.required_columns = (
            self.apc_features + ["equipment"]
        )

        if len(self.apc_features) != 110:

            raise RuntimeError(
                "Unexpected APC feature count."
            )

        if len(set(self.apc_features)) != 110:

            raise RuntimeError(
                "Duplicate APC feature names."
            )

        if self.input_columns[
            "mean_thickness"
        ] != self.apc_features:

            raise RuntimeError(
                "Invalid thickness input schema."
            )

        if self.input_columns[
            "mean_resistance"
        ] != self.required_columns:

            raise RuntimeError(
                "Invalid resistance input schema."
            )

        if set(
            self.warning_profile["feature_statistics"]
        ) != set(self.apc_features):

            raise RuntimeError(
                "Warning profile schema mismatch."
            )

        dataset_hash = (
            self.metadata["dataset_sha256"]
        )

        development_hash = self.metadata[
            "development_population_sha256"
        ]

        if self.warning_profile[
            "dataset_sha256"
        ] != dataset_hash:

            raise RuntimeError(
                "Warning profile dataset mismatch."
            )

        if self.warning_profile[
            "development_population_sha256"
        ] != development_hash:

            raise RuntimeError(
                "Warning profile population mismatch."
            )

        if self.evaluation[
            "dataset_sha256"
        ] != dataset_hash:

            raise RuntimeError(
                "Evaluation dataset mismatch."
            )

        if self.evaluation[
            "partition_summary"
        ][
            "development_population_sha256"
        ] != development_hash:

            raise RuntimeError(
                "Evaluation population mismatch."
            )

        if self.evaluation[
            "model_configurations"
        ] != self.metadata[
            "model_configurations"
        ]:

            raise RuntimeError(
                "Evaluation configuration mismatch."
            )

        for target in TARGETS:

            expected = self.input_columns[target]

            fitted = list(
                self.models[target].feature_names_in_
            )

            if fitted != expected:

                raise RuntimeError(
                    f"Fitted input schema mismatch: {target}"
                )


    def validate_inputs(self, inputs):

        """
        Validate an inference batch.

        Required columns must be present.
        Missing individual measurements are allowed.

        Numeric strings, booleans and infinity
        are rejected rather than silently coerced.
        """

        if not isinstance(inputs, pd.DataFrame):

            raise InputValidationError(
                "Inputs must be a pandas DataFrame."
            )

        if inputs.empty:

            raise InputValidationError(
                "Input batch cannot be empty."
            )

        if not inputs.columns.is_unique:

            raise InputValidationError(
                "Duplicate input columns detected."
            )

        missing_columns = [
            column

            for column in self.required_columns

            if column not in inputs.columns
        ]

        if missing_columns:

            raise InputValidationError(
                f"Missing required columns: {missing_columns}"
            )

        unexpected_columns = [
            column

            for column in inputs.columns

            if column not in self.required_columns
        ]

        if unexpected_columns:

            raise InputValidationError(
                f"Unexpected input columns: "
                f"{unexpected_columns}"
            )

        validated = inputs[
            self.required_columns
        ].copy()

        validated = validated.reset_index(
            drop=True
        )

        for column in self.apc_features:

            cleaned = []

            for position, value in enumerate(
                validated[column]
            ):

                if value is None or value is pd.NA:

                    cleaned.append(np.nan)

                    continue

                if isinstance(
                    value,
                    (bool, np.bool_),
                ):

                    raise InputValidationError(
                        f"Boolean value in {column}, "
                        f"row {position}."
                    )

                if not isinstance(value, Real):

                    raise InputValidationError(
                        f"Non-numeric value in {column}, "
                        f"row {position}."
                    )

                numeric = float(value)

                if np.isinf(numeric):

                    raise InputValidationError(
                        f"Infinite value in {column}, "
                        f"row {position}."
                    )

                cleaned.append(numeric)

            validated[column] = np.asarray(
                cleaned,
                dtype=float,
            )

        equipment_values = []

        for position, value in enumerate(
            validated["equipment"]
        ):

            if value is None or value is pd.NA:

                equipment_values.append(np.nan)

                continue

            if isinstance(value, Real) and np.isnan(value):

                equipment_values.append(np.nan)

                continue

            if not isinstance(value, str):

                raise InputValidationError(
                    f"Invalid equipment value "
                    f"at row {position}."
                )

            value = value.strip()

            if not value:

                equipment_values.append(np.nan)

            else:

                equipment_values.append(value)

        validated["equipment"] = pd.Series(
            equipment_values,
            dtype=object,
        )

        all_missing = validated[
            self.apc_features
        ].isna().all(axis=1)

        if all_missing.any():

            positions = np.flatnonzero(
                all_missing.to_numpy()
            ).tolist()

            raise InputValidationError(
                "All APC measurements are missing "
                f"for rows: {positions}"
            )

        return validated


    def _predict_validated(self, validated):

        predictions = {}

        for target in TARGETS:

            columns = self.input_columns[target]

            values = np.asarray(
                self.models[target].predict(
                    validated[columns]
                ),
                dtype=float,
            )

            if values.shape != (len(validated),):

                raise PredictionError(
                    f"Unexpected output shape: {target}"
                )

            if not np.isfinite(values).all():

                raise PredictionError(
                    f"Non-finite model predictions: {target}"
                )

            if (values <= 0).any():

                raise PredictionError(
                    f"Non-positive model predictions: {target}"
                )

            predictions[target] = values

        return predictions


    @staticmethod
    def _format_warnings(row):

        warning_codes = []

        if row["Elevated Missingness"]:

            warning_codes.append(
                "ELEVATED_MISSINGNESS"
            )

        if row["Extreme Sensor Warning"]:

            warning_codes.append(
                "EXTREME_SENSOR_VALUES"
            )

        if row["Unknown Equipment"]:

            warning_codes.append(
                "UNKNOWN_EQUIPMENT"
            )

        if row["Missing Equipment"]:

            warning_codes.append(
                "MISSING_EQUIPMENT"
            )

        return {
            "has_warning": bool(
                row["Any Input Warning"]
            ),

            "codes": warning_codes,

            "missing_apc_inputs": int(
                row["Missing APC Inputs"]
            ),

            "extreme_apc_inputs": int(
                row["Extreme APC Inputs"]
            ),

            "constant_feature_deviations": int(
                row["Constant Feature Deviations"]
            ),

            "previously_unobserved_features": int(
                row["Previously Unobserved Features"]
            ),

            "calibrated_uncertainty": False,
        }


    def predict_batch(self, inputs):

        """
        Generate both material-property predictions.

        Return a JSON-compatible batch response.
        """

        validated = self.validate_inputs(
            inputs
        )

        predictions = self._predict_validated(
            validated
        )

        warnings = calculate_input_warnings(
            validated,
            self.warning_profile,
        )

        records = []

        for position in range(len(validated)):

            equipment = validated.iloc[
                position
            ]["equipment"]

            records.append({
                "row_index": position,

                "equipment": (
                    None

                    if pd.isna(equipment)

                    else str(equipment)
                ),

                "predictions": {
                    target: float(
                        predictions[target][position]
                    )

                    for target in TARGETS
                },

                "warnings": self._format_warnings(
                    warnings.iloc[position]
                ),
            })

        return {
            "model_version": self.model_version,

            "record_count": len(records),

            "output_scale": "dataset_units",

            "records": records,

            "interpretation": (
                "Virtual metrology estimates for "
                "engineering review; not wafer "
                "acceptance decisions."
            ),
        }


    def predict_one(self, inputs):

        """
        Generate predictions for exactly one record.
        """

        if not isinstance(inputs, dict):

            raise InputValidationError(
                "Single-record input must be a dictionary."
            )

        return self.predict_batch(
            pd.DataFrame([inputs])
        )


    def explain_one(
        self,
        inputs,
        top_n=10,
    ):

        """
        Generate grounded dual-target local
        explanations using the final fitted models.
        """

        if not isinstance(inputs, dict):

            raise InputValidationError(
                "Single-record input must be a dictionary."
            )

        if (
            isinstance(top_n, bool)

            or not isinstance(top_n, int)

            or not 1 <= top_n <= 20
        ):

            raise InputValidationError(
                "top_n must be an integer from 1 to 20."
            )

        validated = self.validate_inputs(
            pd.DataFrame([inputs])
        )

        response = self.predict_batch(
            validated
        )

        prediction_record = response[
            "records"
        ][0]

        explanations = {}

        for target in TARGETS:

            explanation = explain_single_prediction(
                model=self.models[target],

                inputs=validated[
                    self.input_columns[target]
                ],

                apc_features=self.apc_features,

                target=target,

                top_n=top_n,
            )

            if not np.isclose(
                explanation["prediction"],

                prediction_record[
                    "predictions"
                ][target],

                rtol=1e-6,

                atol=1e-10,
            ):

                raise PredictionError(
                    f"Explanation prediction mismatch: {target}"
                )

            explanations[target] = explanation

        return {
            **response,

            "explanations": explanations,

            "interpretation_restrictions": [
                "Feature contributions describe fitted "
                "model behaviour, not physical causality.",

                "Input warnings are heuristic rather "
                "than calibrated uncertainty estimates.",

                "No engineering acceptance thresholds "
                "are inferred from these predictions.",

                "Physical metrology remains necessary "
                "for measured quality assessment.",
            ],
        }


    def get_model_information(self):

        """
        Return frontend-safe model information.
        """

        return {
            "model_version": self.model_version,

            "targets": list(TARGETS),

            "metadata": self.metadata,

            "input_schema": self.input_schema,

            "final_performance": self.evaluation[
                "selected_model_comparison"
            ],

            "global_explainability": (
                self.global_explainability
            ),
        }