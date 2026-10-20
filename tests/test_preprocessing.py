"""Tests for handle_missing_values — threshold-based missing value handling
(drop column >50%, drop rows 30-50%, fill+indicator 10-30%, fill <10%)."""
import numpy as np
import pandas as pd

from utils.preprocessing import *


def test_column_not_found_returns_unchanged():
    """Should return df unchanged and report column_not_found if the
    column doesn't exist."""
    df = pd.DataFrame({"a": [1, 2, 3]})

    result_df, metadata = handle_missing_values(df, "nonexistent")

    assert "nonexistent" not in result_df.columns
    assert metadata["action"] == "column_not_found"
    assert metadata["missing_count"] is None


def test_empty_dataframe_returns_unchanged():
    """Should handle an empty DataFrame gracefully."""
    df = pd.DataFrame({"a": []})

    result_df, metadata = handle_missing_values(df, "a")

    assert metadata["action"] == "empty_dataframe"
    assert len(result_df) == 0


def test_drops_column_when_missing_exceeds_50_percent():
    """>50% missing should drop the entire column."""
    df = pd.DataFrame({
        "a": [1, np.nan, np.nan, np.nan, np.nan],  # 80% missing
        "b": [1, 2, 3, 4, 5],
    })

    result_df, metadata = handle_missing_values(df, "a")

    assert "a" not in result_df.columns
    assert "b" in result_df.columns  # untouched
    assert metadata["action"] == "drop_column"
    assert metadata["missing_percentage"] == 80.0


def test_drops_rows_when_missing_between_30_and_50_percent():
    """30-50% missing should drop only the rows missing that column."""
    df = pd.DataFrame({
        "a": [1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, 10],  # 20% missing... need >30%
    })
    # Build exact 40% missing (4 of 10)
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6, np.nan, np.nan, np.nan, np.nan],
    })

    result_df, metadata = handle_missing_values(df, "a")

    assert result_df["a"].isna().sum() == 0
    assert len(result_df) == 6  # 10 - 4 dropped
    assert metadata["action"] == "drop_rows"
    assert metadata["rows_dropped"] == 4
    # index should be reset
    assert list(result_df.index) == list(range(len(result_df)))


def test_fills_with_indicator_when_missing_between_10_and_30_percent_numeric():
    """10-30% missing on a numeric column should fill with mean and add
    an indicator column."""
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, np.nan],  # 20% missing
    })

    result_df, metadata = handle_missing_values(df, "a")

    assert result_df["a"].isna().sum() == 0
    assert "a_was_missing" in result_df.columns
    assert result_df["a_was_missing"].sum() == 2
    assert metadata["action"] == "fill_and_indicator"
    assert metadata["fill_strategy"] == "mean"
    assert metadata["indicator_column"] == "a_was_missing"


def test_fills_with_indicator_when_missing_between_10_and_30_percent_categorical():
    """10-30% missing on a non-numeric column should fill with mode and
    add an indicator column."""
    df = pd.DataFrame({
        "cat": ["x", "x", "x", "y", "x", "x", "x", "x", None, None],  # 20% missing
    })

    result_df, metadata = handle_missing_values(df, "cat")

    assert result_df["cat"].isna().sum() == 0
    assert "cat_was_missing" in result_df.columns
    assert metadata["fill_strategy"] == "mode"
    assert metadata["fill_value"] == "x"  # most frequent value


def test_fills_only_when_missing_below_10_percent_numeric():
    """<10% missing on a numeric column should fill with median, no
    indicator column added."""
    df = pd.DataFrame({
        "a": list(range(1, 20)) + [np.nan],  # ~5% missing (1 of 20)
    })

    result_df, metadata = handle_missing_values(df, "a")

    assert result_df["a"].isna().sum() == 0
    assert "a_was_missing" not in result_df.columns
    assert metadata["action"] == "fill"
    assert metadata["fill_strategy"] == "median"


def test_fills_only_when_missing_below_10_percent_categorical():
    """<10% missing on a non-numeric column should fill with mode, no
    indicator column added."""
    df = pd.DataFrame({
        "cat": ["x"] * 19 + [None],  # 5% missing
    })

    result_df, metadata = handle_missing_values(df, "cat")

    assert result_df["cat"].isna().sum() == 0
    assert "cat_was_missing" not in result_df.columns
    assert metadata["fill_value"] == "x"


def test_no_missing_values_reports_correctly():
    """A column with zero missing values should be left untouched and
    reported as no_missing_values."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})

    result_df, metadata = handle_missing_values(df, "a")

    assert metadata["action"] == "no_missing_values"
    assert metadata["missing_percentage"] == 0.0
    pd.testing.assert_frame_equal(result_df, df)


def test_fill_value_is_json_serializable():
    """fill_value should be a plain Python type (via .item()), not a
    numpy scalar, since this feeds into JSON artifact logging later."""
    df = pd.DataFrame({
        "a": list(range(1, 20)) + [np.nan],
    })

    _, metadata = handle_missing_values(df, "a")

    assert isinstance(metadata["fill_value"], (int, float))
    assert not isinstance(metadata["fill_value"], np.generic)

def test_produces_expected_dummy_columns():
    """Should produce one dummy column per category, minus the dropped
    baseline, each prefixed with the original column name."""
    df = pd.DataFrame({"gender": ["Male", "Female", "Other", "Male"]})

    dummies, metadata = _encode_column_ohe(df, "gender")

    # Female is dropped (alphabetically first), Male and Other remain
    assert set(dummies.columns) == {"gender_Male", "gender_Other"}
    assert metadata["encoded_columns"] == list(dummies.columns)


def test_dropped_baseline_is_first_alphabetical_category():
    """The dropped baseline should be the first category in sorted order,
    matching pd.get_dummies(drop_first=True) behavior."""
    df = pd.DataFrame({"gender": ["Other", "Male", "Female"]})

    _, metadata = _encode_column_ohe(df, "gender")

    assert metadata["dropped_baseline_category"] == "Female"
    assert metadata["all_categories"] == ["Female", "Male", "Other"]


def test_drop_first_false_keeps_all_categories():
    """With drop_first=False, every category should get its own dummy
    column, and dropped_baseline_category should be None."""
    df = pd.DataFrame({"gender": ["Male", "Female", "Other"]})

    dummies, metadata = _encode_column_ohe(df, "gender", drop_first=False)

    assert set(dummies.columns) == {"gender_Female", "gender_Male", "gender_Other"}
    assert metadata["dropped_baseline_category"] is None


def test_nan_values_excluded_from_categories_but_row_preserved():
    """NaN values shouldn't appear in all_categories, but the row itself
    should still exist in the output (all-zero across dummy columns)."""
    df = pd.DataFrame({"gender": ["Male", None, "Female"]})

    dummies, metadata = _encode_column_ohe(df, "gender")

    assert None not in metadata["all_categories"]
    assert len(dummies) == 3  # row count preserved
    # the NaN row should be all zeros across dummy columns
    assert dummies.iloc[1].sum() == 0


def test_dummy_values_are_int_not_bool():
    """Dummy columns should be int dtype (0/1), not bool, per dtype=int
    explicitly passed to get_dummies."""
    df = pd.DataFrame({"gender": ["Male", "Female"]})

    dummies, _ = _encode_column_ohe(df, "gender")

    for col in dummies.columns:
        assert dummies[col].dtype == int or dummies[col].dtype == "int64"


def test_row_values_correctly_reflect_original_category():
    """Each row's dummy values should correctly one-hot-encode its
    original category — the actual correctness check, not just shape."""
    df = pd.DataFrame({"gender": ["Male", "Other", "Female"]})

    dummies, _ = _encode_column_ohe(df, "gender")

    assert dummies.loc[0, "gender_Male"] == 1
    assert dummies.loc[0, "gender_Other"] == 0

    assert dummies.loc[1, "gender_Male"] == 0
    assert dummies.loc[1, "gender_Other"] == 1

    assert dummies.loc[2, "gender_Male"] == 0
    assert dummies.loc[2, "gender_Other"] == 0  # Female = dropped baseline


def test_single_category_column_with_drop_first():
    """If a column has only one category, drop_first=True should result
    in zero dummy columns and that category as the dropped baseline."""
    df = pd.DataFrame({"gender": ["Male", "Male", "Male"]})

    dummies, metadata = _encode_column_ohe(df, "gender")

    assert len(dummies.columns) == 0
    assert metadata["dropped_baseline_category"] == "Male"


def test_all_categories_sorted():
    """all_categories should be sorted, not in first-seen order, since
    this is what determines which category becomes the dropped baseline."""
    df = pd.DataFrame({"gender": ["Other", "Female", "Male", "Other"]})

    _, metadata = _encode_column_ohe(df, "gender")

    assert metadata["all_categories"] == sorted(metadata["all_categories"])
