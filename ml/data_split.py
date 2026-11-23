
"""
STRATUM - Chronological and Lot-Aware Data Splitting

Creates training, validation, and test partitions.

Rules:
- Respect processing-time order.
- Prevent production-lot overlap.
- Keep held-out test records separate.
- Do not perform preprocessing or model training here.
"""

import pandas as pd


def chronological_lot_split(
    df,
    test_fraction=0.20,
    validation_fraction=0.20,
):
    """
    Split records chronologically.

    validation_fraction is measured relative to
    the development partition, not the full dataset.

    At a boundary, if a lot appears in both partitions,
    its earlier-partition records are purged.

    Returns:
        train_df, val_df, test_df, split_info
    """

    required_columns = [
        "runstart",
        "wafer_id",
        "lot",
        "mean_thickness",
        "mean_resistance",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Required columns missing: {missing}"
        )

    if not 0 < test_fraction < 1:
        raise ValueError("Invalid test fraction.")

    if not 0 < validation_fraction < 1:
        raise ValueError("Invalid validation fraction.")

    if df["lot"].isna().any():
        raise ValueError("Missing production-lot identifiers.")

    if df["wafer_id"].duplicated().any():
        raise ValueError("Duplicate wafer IDs found.")

    data = df.copy()

    data["runstart"] = pd.to_datetime(
        data["runstart"],
        errors="coerce",
    )

    if data["runstart"].isna().any():
        raise ValueError("Invalid processing timestamps.")

    data = data.sort_values(
        ["runstart", "wafer_id"]
    ).reset_index(drop=True)

    # First boundary: development vs test
    test_start = int(
        len(data) * (1 - test_fraction)
    )

    development = data.iloc[:test_start].copy()
    test = data.iloc[test_start:].copy()

    # Remove any development records sharing test lots
    test_lots = set(test["lot"])

    development_overlap = development["lot"].isin(
        test_lots
    )

    purged_dev = int(development_overlap.sum())

    development = development.loc[
        ~development_overlap
    ].copy()

    # Second boundary: training vs validation
    val_start = int(
        len(development) * (1 - validation_fraction)
    )

    train = development.iloc[:val_start].copy()
    val = development.iloc[val_start:].copy()

    # Remove training records sharing validation lots
    val_lots = set(val["lot"])

    train_overlap = train["lot"].isin(val_lots)

    purged_train = int(train_overlap.sum())

    train = train.loc[
        ~train_overlap
    ].copy()

    if any(part.empty for part in [train, val, test]):
        raise ValueError("A data partition is empty.")

    # Ensure lot independence
    assert set(train["lot"]).isdisjoint(val["lot"])
    assert set(train["lot"]).isdisjoint(test["lot"])
    assert set(val["lot"]).isdisjoint(test["lot"])

    # Ensure chronological separation
    assert train["runstart"].max() <= val["runstart"].min()
    assert val["runstart"].max() <= test["runstart"].min()

    # Ensure records were not duplicated or silently lost
    all_ids = pd.concat([
        train["wafer_id"],
        val["wafer_id"],
        test["wafer_id"],
    ])

    assert all_ids.is_unique

    assert (
        len(train) + len(val) + len(test)
        + purged_dev + purged_train
        == len(data)
    )

    split_info = {
        "total_records": len(data),
        "train_records": len(train),
        "validation_records": len(val),
        "test_records": len(test),
        "purged_development_records": purged_dev,
        "purged_training_records": purged_train,
        "train_start": str(train["runstart"].min()),
        "train_end": str(train["runstart"].max()),
        "validation_start": str(val["runstart"].min()),
        "validation_end": str(val["runstart"].max()),
        "test_start": str(test["runstart"].min()),
        "test_end": str(test["runstart"].max()),
    }

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
        split_info,
    )