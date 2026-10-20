"""Tests for train_model, evaluate_model and tqdm_joblib."""
import io
import pytest
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression, Ridge
from utils.modeling import evaluate_model, tqdm_joblib, train_model


class StubModel:
    """Returns fixed predictions and records the columns it was given."""

    def __init__(self, preds):
        self.preds = np.asarray(preds)
        self.seen_columns = None

    def predict(self, X):
        self.seen_columns = list(X.columns)
        return self.preds


def _df(y):
    return pd.DataFrame({"f1": range(len(y)), "target": y})


@pytest.fixture
def regression_data():
    """Noise-free linear data, split into (train, val)."""
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 2))
    df = pd.DataFrame(X, columns=["f1", "f2"])
    df["target"] = 3 * X[:, 0] - 2 * X[:, 1] + 1
    return df.iloc[:40].reset_index(drop=True), df.iloc[40:].reset_index(drop=True)


# ---------- train_model ----------

def test_train_without_grid_search_returns_expected_structure(regression_data):
    train_df, val_df = regression_data

    result = train_model(train_df, val_df, LinearRegression(), "target")

    assert set(result) == {"model", "best_params", "metrics"}
    assert result["best_params"] == {}
    assert set(result["metrics"]) == {"mae", "rmse", "r2"}
    # perfectly linear data -> near-perfect fit
    assert result["metrics"]["r2"] == pytest.approx(1.0, abs=1e-6)
    assert result["metrics"]["mae"] == pytest.approx(0.0, abs=1e-6)


def test_target_column_excluded_from_training_features(regression_data):
    train_df, val_df = regression_data

    result = train_model(train_df, val_df, LinearRegression(), "target")

    assert list(result["model"].feature_names_in_) == ["f1", "f2"]


def test_grid_search_returns_best_params_from_grid(regression_data):
    train_df, val_df = regression_data
    grid = {"alpha": [0.01, 1.0, 100.0]}

    result = train_model(train_df, val_df, Ridge(), "target", grid_search_params=grid)

    assert set(result["best_params"]) == {"alpha"}
    assert result["best_params"]["alpha"] == 0.01  # noise-free data -> least regularization wins
    assert result["model"].alpha == result["best_params"]["alpha"]


def test_missing_target_column_raises(regression_data):
    train_df, val_df = regression_data

    with pytest.raises(KeyError):
        train_model(train_df, val_df, LinearRegression(), "nonexistent")


def test_train_model_does_not_mutate_inputs(regression_data):
    train_df, val_df = regression_data
    train_copy, val_copy = train_df.copy(), val_df.copy()

    train_model(train_df, val_df, LinearRegression(), "target")

    pd.testing.assert_frame_equal(train_df, train_copy)
    pd.testing.assert_frame_equal(val_df, val_copy)


# ---------- evaluate_model ----------

def test_regression_metrics_are_correct():
    model = StubModel([1, 2, 3, 6])  # one error of 2

    result = evaluate_model(model, _df([1, 2, 3, 4]), "target", task_type="regression")

    m = result["metrics"]
    assert m["mae"] == pytest.approx(0.5)
    assert m["mse"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(1.0)
    assert m["r2"] == pytest.approx(0.2)
    assert all(isinstance(v, float) for v in m.values())  # MLflow/JSON-safe


def test_classification_metrics_are_correct():
    # y_true=[0,0,1,1], preds=[0,1,1,1]
    model = StubModel([0, 1, 1, 1])

    result = evaluate_model(model, _df([0, 0, 1, 1]), "target", task_type="classification")

    m = result["metrics"]
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(5 / 6)
    assert m["recall"] == pytest.approx(0.75)
    assert m["f1_score"] == pytest.approx(11 / 15)
    assert all(isinstance(v, float) for v in m.values())


@pytest.mark.filterwarnings("error::sklearn.exceptions.UndefinedMetricWarning")
def test_classification_handles_class_never_predicted():
    """zero_division=0 should prevent warnings/crashes when a class gets no predictions."""
    model = StubModel([0, 0])

    result = evaluate_model(model, _df([0, 1]), "target", task_type="classification")

    assert result["metrics"]["precision"] == pytest.approx(0.25)
    assert result["metrics"]["recall"] == pytest.approx(0.5)


def test_invalid_task_type_raises():
    with pytest.raises(ValueError, match="task_type"):
        evaluate_model(StubModel([1, 2]), _df([1, 2]), "target", task_type="clustering")


def test_target_column_not_passed_to_model():
    model = StubModel([1, 2])

    evaluate_model(model, _df([1, 2]), "target", task_type="regression")

    assert model.seen_columns == ["f1"]


def test_returns_flat_arrays_even_for_2d_predictions():
    model = StubModel([[1], [2], [3]])  # column-vector output

    result = evaluate_model(model, _df([1, 2, 3]), "target", task_type="regression")

    assert result["predictions"].ndim == 1
    assert result["y_true"].ndim == 1
    assert len(result["predictions"]) == len(result["y_true"]) == 3


# ---------- tqdm_joblib ----------

def test_tqdm_joblib_restores_original_callback_even_on_error():
    original = joblib.parallel.BatchCompletionCallBack

    with pytest.raises(RuntimeError):
        with tqdm_joblib(tqdm(total=1, file=io.StringIO())):
            raise RuntimeError("boom")

    assert joblib.parallel.BatchCompletionCallBack is original


def test_tqdm_joblib_updates_progress_bar():
    bar = tqdm(total=4, file=io.StringIO())

    with tqdm_joblib(bar):
        Parallel(n_jobs=2, backend="threading")(delayed(abs)(-i) for i in range(4))

    assert bar.n == 4
    