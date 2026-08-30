"""Model training utilities."""
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier


def split_data(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.70,
    val_test_split: float = 0.50,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a DataFrame into stratified train/val/test sets.

    Args:
        df: The full processed DataFrame.
        target_col: Name of the target column.
        train_size: Fraction of data used for training (remainder is split
            into val/test).
        val_test_split: Fraction of the non-train remainder used for
            validation (the rest goes to test).
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_size), random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_split, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgb_with_gridsearch(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict,
    cv: int = 5,
    random_state: int = 42,
):
    """Manually search a parameter grid for the best XGBoost configuration,
    showing a single tqdm progress bar (one tick per candidate).

    Args:
        X_train: Training features.
        y_train: Training labels.
        param_grid: Hyperparameter grid to search.
        cv: Number of cross-validation folds per candidate.
        random_state: Random seed for reproducibility.

    Returns:
        A dict with keys "best_estimator_", "best_params_", "best_score_".
    """
    candidates = list(ParameterGrid(param_grid))
    best_score = -np.inf
    best_params = None

    for params in tqdm(candidates, desc="Grid search"):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            **params,
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_score = scores.mean()

        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    best_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        **best_params,
    )
    best_model.fit(X_train, y_train)

    return {
        "best_estimator_": best_model,
        "best_params_": best_params,
        "best_score_": best_score,
    }
