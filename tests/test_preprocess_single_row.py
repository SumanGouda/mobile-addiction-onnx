"""Tests for preprocess_single_row: cleaning one raw input dict using the
saved encoding and fill-missing metadata."""
import pandas as pd
import pytest

from utils.preprocessing import preprocess_single_row


@pytest.fixture
def encoding_meta():
    return {
        "gender": {"metadata": {
            "all_categories": ["Female", "Male", "Other"],
            "encoded_columns": ["gender_Male", "gender_Other"],
        }},
        "stress_level": {"metadata": {
            "all_categories": ["High", "Low", "Medium"],
            "encoded_columns": ["stress_level_Low", "stress_level_Medium"],
        }},
        "academic_work_impact": {"metadata": {
            "all_categories": ["No", "Yes"],
            "encoded_columns": ["academic_work_impact_Yes"],
        }},
    }


@pytest.fixture
def fill_meta():
    return {
        "age": {"fill_value": 27.0, "indicator_column": None},
        "daily_screen_time_hours": {
            "fill_value": 7.64,
            "indicator_column": "daily_screen_time_hours_was_missing",
        },
        "addicted_label": {"fill_value": None, "indicator_column": None},
        "gender_Male": {"fill_value": None, "indicator_column": None},
    }


@pytest.fixture
def raw():
    return {
        "age": 21.0,
        "daily_screen_time_hours": 5.0,
        "gender": "Male",
        "stress_level": "Low",
        "academic_work_impact": "Yes",
    }


# ---------- encoding ----------

def test_known_categories_are_one_hot_encoded(raw, encoding_meta, fill_meta):
    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert result.loc[0, "gender_Male"] == 1
    assert result.loc[0, "gender_Other"] == 0
    assert result.loc[0, "stress_level_Low"] == 1
    assert result.loc[0, "stress_level_Medium"] == 0
    assert result.loc[0, "academic_work_impact_Yes"] == 1


def test_original_categorical_columns_are_removed(raw, encoding_meta, fill_meta):
    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    for col in ("gender", "stress_level", "academic_work_impact"):
        assert col not in result.columns


def test_baseline_categories_give_all_zero_dummies(raw, encoding_meta, fill_meta):
    """Female / High / No were dropped during training, so they must encode as all zeros."""
    raw.update({"gender": "Female", "stress_level": "High", "academic_work_impact": "No"})

    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    dummy_cols = ["gender_Male", "gender_Other", "stress_level_Low",
                  "stress_level_Medium", "academic_work_impact_Yes"]
    assert (result[dummy_cols] == 0).all().all()


def test_unseen_category_raises(raw, encoding_meta, fill_meta):
    raw["gender"] = "Unknown"

    with pytest.raises(ValueError, match="unseen category"):
        preprocess_single_row(raw, encoding_meta, fill_meta)


def test_missing_categorical_value_becomes_all_zero(raw, encoding_meta, fill_meta):
    raw["gender"] = None

    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert result.loc[0, "gender_Male"] == 0
    assert result.loc[0, "gender_Other"] == 0


def test_absent_categorical_key_becomes_all_zero(raw, encoding_meta, fill_meta):
    del raw["stress_level"]

    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert result.loc[0, "stress_level_Low"] == 0
    assert result.loc[0, "stress_level_Medium"] == 0


def test_dummy_columns_are_integers(raw, encoding_meta, fill_meta):
    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    for col in ("gender_Male", "gender_Other", "stress_level_Low"):
        assert pd.api.types.is_integer_dtype(result[col])


# ---------- missing-value filling ----------

def test_present_values_unchanged_and_indicator_is_zero(raw, encoding_meta, fill_meta):
    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert result.loc[0, "age"] == 21.0
    assert result.loc[0, "daily_screen_time_hours"] == 5.0
    assert result.loc[0, "daily_screen_time_hours_was_missing"] == 0


@pytest.mark.parametrize("missing_value", [None, float("nan")])
def test_missing_numeric_filled_with_saved_value_and_indicator_set(
    raw, encoding_meta, fill_meta, missing_value
):
    raw["daily_screen_time_hours"] = missing_value

    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert result.loc[0, "daily_screen_time_hours"] == pytest.approx(7.64)
    assert result.loc[0, "daily_screen_time_hours_was_missing"] == 1


def test_absent_numeric_key_is_treated_as_missing(raw, encoding_meta, fill_meta):
    del raw["daily_screen_time_hours"]

    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert result.loc[0, "daily_screen_time_hours"] == pytest.approx(7.64)
    assert result.loc[0, "daily_screen_time_hours_was_missing"] == 1


def test_column_without_indicator_is_filled_but_gets_no_indicator(raw, encoding_meta, fill_meta):
    raw["age"] = None

    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert result.loc[0, "age"] == pytest.approx(27.0)
    assert "age_was_missing" not in result.columns


def test_fill_entries_with_no_fill_value_are_skipped(raw, encoding_meta, fill_meta):
    """addicted_label has fill_value=None, so it must not be added to the row."""
    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert "addicted_label" not in result.columns


# ---------- output shape / safety ----------

def test_returns_single_row_dataframe(raw, encoding_meta, fill_meta):
    result = preprocess_single_row(raw, encoding_meta, fill_meta)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1


def test_does_not_mutate_input_dict(raw, encoding_meta, fill_meta):
    original = dict(raw)

    preprocess_single_row(raw, encoding_meta, fill_meta)

    assert raw == original
    