"""Tests for src/models/predict.py — align_features"""
import pandas as pd

from src.models.predict import align_features


def test_columns_reordered_to_match_expected():
    """Should reorder columns to exactly match expected_features order,
    even if input columns are in a different order."""
    df = pd.DataFrame({
        "b": [1, 2],
        "a": [3, 4],
        "c": [5, 6],
    })
    expected_features = ["a", "b", "c"]

    result = align_features(df, expected_features)

    assert list(result.columns) == ["a", "b", "c"]


def test_missing_columns_filled_with_zero():
    """If the model expects a column the input doesn't have (e.g. a
    category missing from this batch), it should be added and filled
    with 0, not silently dropped or left to error later."""
    df = pd.DataFrame({
        "a": [1, 2],
        "b": [3, 4],
    })
    expected_features = ["a", "b", "gender_Other"]

    result = align_features(df, expected_features)

    assert "gender_Other" in result.columns
    assert (result["gender_Other"] == 0).all()


def test_extra_columns_not_in_expected_are_dropped():
    """Columns present in the input but not expected by the model
    (e.g. a leftover 'id' column) should be dropped, not passed through
    to the model."""
    df = pd.DataFrame({
        "id": [101, 102],
        "a": [1, 2],
        "b": [3, 4],
    })
    expected_features = ["a", "b"]

    result = align_features(df, expected_features)

    assert "id" not in result.columns
    assert list(result.columns) == ["a", "b"]


def test_row_values_preserved_correctly_after_alignment():
    """Row values should stay correctly matched to their original data
    after reordering/adding columns — this is the actual danger case:
    silently misaligned values would still 'work' but produce wrong
    predictions."""
    df = pd.DataFrame({
        "b": [10, 20],
        "a": [1, 2],
    })
    expected_features = ["a", "b"]

    result = align_features(df, expected_features)

    assert result.loc[0, "a"] == 1
    assert result.loc[0, "b"] == 10
    assert result.loc[1, "a"] == 2
    assert result.loc[1, "b"] == 20


def test_exact_match_returns_unchanged_values():
    """When input columns already exactly match expected_features, the
    values should pass through unchanged."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    })
    expected_features = ["a", "b"]

    result = align_features(df, expected_features)

    pd.testing.assert_frame_equal(result, df)


def test_row_index_preserved():
    """reindex operates on columns, not rows — row index/order should be
    completely untouched."""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [4, 5, 6],
    }, index=[10, 11, 12])
    expected_features = ["a", "b"]

    result = align_features(df, expected_features)

    assert list(result.index) == [10, 11, 12]
    