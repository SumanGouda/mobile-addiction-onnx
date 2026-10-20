import joblib
import contextlib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from typing import Any, Dict, List, Optional, Literal
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error,
    mean_squared_error, precision_score, r2_score, recall_score,
)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report progress using tqdm bar."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def train_model(train_df: pd.DataFrame, val_df: pd.DataFrame, model: Any, target_column: str, grid_search_params: Optional[Dict[str, List[Any]]] = None,) -> Dict[str, Any]:
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    X_val = val_df.drop(columns=[target_column])
    y_val = val_df[target_column]

    best_params = {}

    if grid_search_params:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params,
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            verbose=0, 
        )
        num_combinations = len(
            list(
                grid_search._get_param_iterator()
                if hasattr(grid_search, "_get_param_iterator")
                else [1]
            )
        )
        total_fits = num_combinations * grid_search.cv

        # Run Grid Search wrapped inside tqdm progress bar
        with tqdm_joblib(
            tqdm(desc="Training GridSearch Models", total=total_fits)
        ):
            grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_

    else:
        # Standard fit with simple tqdm progress indicator
        with tqdm(total=1, desc="Fitting Single Model") as pbar:
            model.fit(X_train, y_train)
            pbar.update(1)
        best_model = model

    predictions = best_model.predict(X_val)
    metrics = {
        "mae": mean_absolute_error(y_val, predictions),
        "rmse": np.sqrt(mean_squared_error(y_val, predictions)),
        "r2": r2_score(y_val, predictions),
    }
    return {
        "model": best_model,
        "best_params": best_params,
        "metrics": metrics,
    }


def evaluate_model(model: Any, test_df: pd.DataFrame, target_column: str, task_type: Literal["regression", "classification"] = "regression",) -> Dict[str, Any]:
    """Evaluates a trained model on a target dataset split.

    Returns metrics dict and prediction arrays compatible with MLflow logging
    and downstream plotting utilities.
    """
    X_test = test_df.drop(columns=[target_column])
    y_test = test_df[target_column].to_numpy().ravel()
    predictions = model.predict(X_test).ravel()

    if task_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(
                precision_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_test, predictions, average="weighted", zero_division=0
                )
            ),
        }

    elif task_type == "regression":
        mse = mean_squared_error(y_test, predictions)
        metrics = {
            "mae": float(mean_absolute_error(y_test, predictions)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test, predictions)),
        }

    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    return {
        "metrics": metrics,
        "y_true": y_test,
        "predictions": predictions,
    }
