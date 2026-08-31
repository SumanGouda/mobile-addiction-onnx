"""Tests for src/features/encoding.py — apply_gender_encoding"""
import json

import pandas as pd
import pytest

from src.features.encoding import apply_gender_encoding


@pytest.fixture
def metadata_file(tmp_path):
    """Writes a sample encoding_metadata.json to a temp path and returns it."""
    metadata = {
        "gender": {
            "method": "one_hot",
            "categories": ["Female", "Male", "Other"],
            "dropped_baseline": "Female",
            "resulting_columns": ["gender_Male", "gender_Other"],
        }
    }
    path = tmp_path / "encoding_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f)
    return str(path)


@pytest.fixture
def metadata_file_with_unknown(tmp_path):
    """Metadata where 'Unknown' was a valid training-time category."""
    metadata = {
        "gender": {
            "method": "one_hot",
            "categories": ["Female", "Male", "Other", "Unknown"],
            "dropped_baseline": "Female",
            "resulting_columns": ["gender_Male", "gender_Other", "gender_Unknown"],
        }
    }
    path = tmp_path / "encoding_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f)
    return str(path)


def test_produces_expected_columns_matching_metadata(metadata_file):
    """Should produce exactly the gender_* columns recorded in training
    metadata, even if this batch doesn't contain every category."""
    df = pd.DataFrame({"gender": ["Male", "Male", "Female"]})

    result = apply_gender_encoding(df, metadata_path=metadata_file)

    assert "gender_Male" in result.columns
    assert "gender_Other" in result.columns
    assert "gender" not in result.columns  # original column replaced


def test_missing_category_in_batch_filled_with_zero(metadata_file):
    """If this batch has no 'Other' rows at all, gender_Other should
    still exist (from metadata) and be all zeros, not silently dropped."""
    df = pd.DataFrame({"gender": ["Male", "Male", "Female"]})  # no 'Other' present

    result = apply_gender_encoding(df, metadata_path=metadata_file)

    assert result["gender_Other"].sum() == 0
    assert result["gender_Other"].dtype == int or result["gender_Other"].dtype == "int64"


def test_encoding_values_are_correct_per_row(metadata_file):
    """Each row's dummy columns should correctly reflect its original
    gender value."""
    df = pd.DataFrame({"gender": ["Male", "Other", "Female"]})

    result = apply_gender_encoding(df, metadata_path=metadata_file)

    assert result.loc[0, "gender_Male"] == 1
    assert result.loc[0, "gender_Other"] == 0

    assert result.loc[1, "gender_Male"] == 0
    assert result.loc[1, "gender_Other"] == 1

    assert result.loc[2, "gender_Male"] == 0
    assert result.loc[2, "gender_Other"] == 0  # Female = dropped baseline, all zero


def test_dummy_columns_are_int_not_bool(metadata_file):
    """Dummy columns should be cast to int, not left as bool (pandas
    get_dummies default in recent versions)."""
    df = pd.DataFrame({"gender": ["Male", "Female"]})

    result = apply_gender_encoding(df, metadata_path=metadata_file)

    assert result["gender_Male"].dtype in (int, "int64", "int32")


def test_nan_filled_as_unknown_when_unknown_was_a_training_category(metadata_file_with_unknown):
    """If 'Unknown' was a valid category during training, missing gender
    values in new data should be filled as 'Unknown', not left as NaN."""
    df = pd.DataFrame({"gender": ["Male", None, "Female"]})

    result = apply_gender_encoding(df, metadata_path=metadata_file_with_unknown)

    assert "gender_Unknown" in result.columns
    assert result.loc[1, "gender_Unknown"] == 1


def test_nan_left_as_all_zero_when_unknown_not_a_training_category(metadata_file):
    """If 'Unknown' was never a training category, a missing gender value
    should silently become an all-zero row (matching the dropped baseline
    behavior) rather than raising an error."""
    df = pd.DataFrame({"gender": ["Male", None, "Female"]})

    result = apply_gender_encoding(df, metadata_path=metadata_file)

    assert result.loc[1, "gender_Male"] == 0
    assert result.loc[1, "gender_Other"] == 0
    