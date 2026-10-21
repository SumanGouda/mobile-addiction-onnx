"""Tests for get_model_feature_order: reading expected feature names,
in training order, from different model types."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from utils.preprocessing import get_model_feature_order


# ---------- stubs: each exposes one library's attribute and nothing else ----------

class SklearnStub:
    def __init__(self, names):
        self.feature_names_in_ = np.array(names)


class CatBoostStub:
    def __init__(self, names):
        self.feature_names_ = names


class LightGBMStub:
    def __init__(self, names):
        self.feature_name_ = names


class BoosterStub:
    """Raw xgboost.Booster: has .feature_names, no get_booster()."""
    def __init__(self, names):
        self.feature_names = names


class XGBWrapperStub:
    """XGBoost sklearn wrapper without feature_names_in_: reached via get_booster()."""
    def __init__(self, names):
        self._booster = BoosterStub(names)

    def get_booster(self):
        return self._booster


class _Schema:
    def __init__(self, names):
        self._names = names

    def input_names(self):
        return self._names


class _Metadata:
    def __init__(self, signature, names):
        self.signature = signature
        self._names = names

    def get_input_schema(self):
        return _Schema(self._names)


class PyFuncStub:
    def __init__(self, names, has_signature=True):
        self.metadata = _Metadata(object() if has_signature else None, names)


FEATURES = ["age", "sleep_hours", "gender_Male"]


# ---------- one test per supported source ----------

@pytest.mark.parametrize("model", [
    SklearnStub(FEATURES),
    CatBoostStub(FEATURES),
    LightGBMStub(FEATURES),
    BoosterStub(FEATURES),
    XGBWrapperStub(FEATURES),
    PyFuncStub(FEATURES),
])
def test_returns_feature_names_in_training_order(model):
    assert get_model_feature_order(model) == FEATURES


def test_returns_a_plain_list_of_strings():
    result = get_model_feature_order(SklearnStub(FEATURES))

    assert isinstance(result, list)
    assert all(isinstance(name, str) for name in result)


# ---------- lookup priority and fall-through ----------

def test_sklearn_attribute_takes_priority_over_others():
    class Both:
        feature_names_in_ = np.array(["first"])
        feature_names_ = ["second"]

    assert get_model_feature_order(Both()) == ["first"]


def test_empty_catboost_names_fall_through_to_next_source():
    class Falls:
        feature_names_ = []
        feature_name_ = ["from_lightgbm"]

    assert get_model_feature_order(Falls()) == ["from_lightgbm"]


# ---------- failure cases ----------

def test_raises_when_booster_has_no_feature_names():
    with pytest.raises(ValueError, match="Could not read feature names"):
        get_model_feature_order(BoosterStub(None))


def test_raises_when_pyfunc_has_no_signature():
    with pytest.raises(ValueError, match="Could not read feature names"):
        get_model_feature_order(PyFuncStub(FEATURES, has_signature=False))


def test_raises_for_object_with_no_known_attributes():
    with pytest.raises(ValueError):
        get_model_feature_order(object())


# ---------- real models ----------

@pytest.fixture
def train_frame():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=FEATURES)
    y = (X["age"] > 0).astype(int)
    return X, y


def test_real_sklearn_model_trained_on_dataframe(train_frame):
    X, y = train_frame
    model = LogisticRegression().fit(X, y)

    assert get_model_feature_order(model) == FEATURES


def test_real_xgboost_model_trained_on_dataframe(train_frame):
    X, y = train_frame
    model = XGBClassifier(n_estimators=2).fit(X, y)

    assert get_model_feature_order(model) == FEATURES


def test_real_xgboost_model_trained_on_numpy_raises(train_frame):
    """No names stored -> must raise, not return an empty or guessed list."""
    X, y = train_frame
    model = XGBClassifier(n_estimators=2).fit(X.to_numpy(), y)

    with pytest.raises(ValueError):
        get_model_feature_order(model)


def test_real_sklearn_model_trained_on_numpy_raises(train_frame):
    X, y = train_frame
    model = LogisticRegression().fit(X.to_numpy(), y)

    with pytest.raises(ValueError):
        get_model_feature_order(model)
        