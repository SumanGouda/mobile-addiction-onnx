"""Tests for src/data/cleaning.py"""
import numpy as np
import pandas as pd
import pytest

from src.data.cleaning import fill_missing_by_correlation


def test_fills_missing_values_when_correlated_predictor_exists():
    """Should fill missing values using a correlated numeric column."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, np.nan, 6, 7, 8, 9, 10],
        "b": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],  # b = 2*a, perfectly correlated
    })

    result_df, msg = fill_missing_by_correlation(df, "a", threshold=0.3)

    assert result_df["a"].isna().sum() == 0
    assert "Filled" in msg
    assert result_df.loc[4, "a"] == pytest.approx(5, abs=0.5)


def test_skips_object_dtype_column():
    """Should skip and not modify string/object columns."""
    df = pd.DataFrame({
        "category": ["low", "medium", None, "high"],
        "numeric": [1, 2, 3, 4],
    })

    result_df, msg = fill_missing_by_correlation(df, "category")

    assert result_df["category"].isna().sum() == 1  # unchanged
    assert "not numeric" in msg or "object/string column" in msg


def test_skips_when_missing_exceeds_threshold():
    """Should skip filling if missing % exceeds missing_threshold."""
    df = pd.DataFrame({
        "a": [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
        "b": [1, 2, 3, 4, 5],
    })

    result_df, msg = fill_missing_by_correlation(df, "a", missing_threshold=0.6)

    assert result_df["a"].isna().sum() == 4  # unchanged
    assert "skipped" in msg
    assert "missing" in msg


def test_skips_when_no_correlated_feature_above_threshold():
    """Should skip and report independence if no column meets the
    correlation threshold."""
    np.random.seed(42)
    df = pd.DataFrame({
        "a": [1, 2, 3, np.nan, 5, 6, 7, 8, 9, 10],
        "unrelated": np.random.permutation(10),
    })

    result_df, msg = fill_missing_by_correlation(df, "a", threshold=0.99)

    assert result_df["a"].isna().sum() == 1  # unchanged
    assert "independent" in msg


def test_skips_when_column_has_no_missing_values():
    """Should report nothing-to-do if the column has no missing values."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": [2, 4, 6, 8, 10],
    })

    result_df, msg = fill_missing_by_correlation(df, "a")

    assert "no missing values" in msg


def test_reports_rows_still_missing_when_predictors_also_null():
    """Should skip and report predictor-unavailability when the only
    rows missing 'a' also have missing predictor values."""
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10],
        "b": [2, 4, np.nan, 8, 10, 12, 14, 16, 18, 20],  # missing at same row as 'a'
    })

    result_df, msg = fill_missing_by_correlation(df, "a", threshold=0.3)

    assert result_df["a"].isna().sum() == 1  # row 2 stays missing
    assert "can't fill via regression" in msg


def test_skips_when_feature_not_in_dataframe():
    """Should skip gracefully if the target feature isn't found among
    numeric columns."""
    df = pd.DataFrame({
        "category": ["a", "b", None, "c"],
    })

    result_df, msg = fill_missing_by_correlation(df, "category")

    # Match the actual return message from fill_missing_by_correlation
    assert "object/string" in msg or "skipped" in msg
    