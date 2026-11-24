
"""
STRATUM - Machine Learning Preprocessing

Defines candidate model inputs and a reusable leakage-safe preprocessing pipeline.

Important:
The preprocessing pipeline must only be fitted after the training/test split.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

# --------------------------------------------
# 1. Dataset metadata
# --------------------------------------------

METADATA_COLUMNS = [
    "bat_id",
    "wafer_id",
    "lot",
    "equipment",
    "machine",
    "batchstart",
    "batchend",
    "slot",
    "runstart",
    "runend",
]


# --------------------------------------------
# 2. Post-process metrology outputs
# --------------------------------------------

THICKNESS_COLUMNS = [
    f"RAW_VALUE_t_{i}" for i in range(1, 18)
]

RESISTANCE_COLUMNS = [
    f"RAW_VALUE_r_{i}" for i in range(1, 18)
]

RESISTIVITY_COLUMNS = [
    f"RAW_VALUE_{i}" for i in range(1, 18)
]

AVERAGE_COLUMNS = [
    "mean_thickness",
    "mean_resistance",
]


# --------------------------------------------
# 3. Quality and stability indicators
# --------------------------------------------

QUALITY_COLUMNS = [
    "stable_state",
    "quality_thickness",
    "quality_resistance",
    "quality_resistivity",
    "quality_overall",
]


# --------------------------------------------
# 4. Permitted and excluded feature groups
# --------------------------------------------

EXCLUDED_COLUMNS = set(
    METADATA_COLUMNS
    + THICKNESS_COLUMNS
    + RESISTANCE_COLUMNS
    + RESISTIVITY_COLUMNS
    + AVERAGE_COLUMNS
    + QUALITY_COLUMNS
)

TARGET_COLUMNS = [
    "mean_thickness",
    "mean_resistance",
]

CATEGORICAL_FEATURES = [
    "equipment",
]


def get_feature_columns(df):
    """
    Identify APC process measurements and append
    permitted categorical equipment metadata.

    The initial feature set contains:
    - 110 numeric APC variables
    - equipment as one categorical feature

    No final feature selection is performed here.
    """

    apc_columns = [
        col for col in df.columns
        if col not in EXCLUDED_COLUMNS
    ]

    if len(apc_columns) != 110:
        raise ValueError(
            "Expected 110 APC variables, found "
            f"{len(apc_columns)}. Verify the CSV schema."
        )

    non_numeric = [
        col for col in apc_columns
        if not df[col].dtype.kind in "biuf"
    ]

    if non_numeric:
        raise ValueError(
            f"Non-numeric APC variables: {non_numeric}"
        )

    feature_columns = (
        apc_columns + CATEGORICAL_FEATURES
    )

    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Duplicate feature names detected.")

    if any(
        col in EXCLUDED_COLUMNS
        for col in apc_columns
    ):
        raise ValueError("Excluded variable detected.")

    return apc_columns, feature_columns


# --------------------------------------------
# 5. Reusable preprocessing pipeline
# --------------------------------------------


def build_preprocessor(
    numeric_features,
    include_equipment=True,
    scale_numeric=False,
):
    """
    Construct an unfitted STRATUM preprocessing pipeline.

    Parameters
    ----------
    numeric_features:
        Numeric APC input column names.

    include_equipment:
        Include one-hot encoded equipment metadata.

    scale_numeric:
        Standardise the imputed numeric feature matrix,
        including missingness indicators.

    All learned transformations must be fitted using
    training data only.
    """

    numeric_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
    ]

    if scale_numeric:
        numeric_steps.append(
            (
                "scaler",
                StandardScaler(),
            )
        )

    numeric_pipeline = Pipeline(
        steps=numeric_steps
    )

    transformers = [
        (
            "numeric",
            numeric_pipeline,
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

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=0,
    )

    return preprocessor
