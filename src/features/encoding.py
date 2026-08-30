"""Categorical feature encoding utilities."""
import json
from typing import Dict, List

import pandas as pd


def one_hot_encode_gender(
    df: pd.DataFrame, save_metadata_path: str = None
) -> pd.DataFrame:
    """One-hot encode the `gender` column, dropping the first alphabetical
    category as baseline. Optionally saves encoding metadata to disk for
    reuse at inference time.

    Args:
        df: DataFrame containing a `gender` column.
        save_metadata_path: If provided, path to write the encoding
            metadata JSON (categories, dropped baseline, resulting columns).

    Returns:
        The DataFrame with `gender` replaced by one-hot columns.
    """
    df = df.copy()
    df["gender"] = df["gender"].fillna("Unknown")

    original_categories = sorted(df["gender"].dropna().unique())
    dropped_category = original_categories[0]

    df = pd.get_dummies(df, columns=["gender"], prefix="gender", drop_first=True)

    gender_cols = [c for c in df.columns if c.startswith("gender_")]
    df[gender_cols] = df[gender_cols].astype(int)

    if save_metadata_path:
        metadata = {
            "gender": {
                "method": "one_hot",
                "categories": original_categories,
                "dropped_baseline": dropped_category,
                "resulting_columns": gender_cols,
            }
        }
        with open(save_metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    return df


def apply_gender_encoding(df: pd.DataFrame, metadata_path: str) -> pd.DataFrame:
    """Apply a previously saved gender one-hot encoding to new data (e.g.
    test/inference data), ensuring identical columns to training.

    Args:
        df: DataFrame containing a `gender` column.
        metadata_path: Path to the encoding metadata JSON saved during
            training.

    Returns:
        The DataFrame with `gender` replaced by one-hot columns matching
        the training schema.
    """
    df = df.copy()
    with open(metadata_path) as f:
        metadata = json.load(f)

    if "Unknown" in metadata["gender"]["categories"]:
        df["gender"] = df["gender"].fillna("Unknown")

    df = pd.get_dummies(df, columns=["gender"], prefix="gender", drop_first=True)

    gender_cols = [c for c in df.columns if c.startswith("gender_")]
    df[gender_cols] = df[gender_cols].astype(int)

    expected_cols = metadata["gender"]["resulting_columns"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    return df


def label_encode_columns(
    df: pd.DataFrame, mappings: Dict[str, Dict[str, int]]
) -> pd.DataFrame:
    """Apply manual label-encoding mappings to one or more columns.

    Args:
        df: The DataFrame to encode.
        mappings: A dict of {column_name: {category: encoded_value}}.

    Returns:
        The DataFrame with the specified columns mapped to integers.
    """
    df = df.copy()
    for col, mapping in mappings.items():
        df[col] = df[col].map(mapping)
    return df
