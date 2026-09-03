"""Model training utilities and model registry."""
from typing import Dict, Tuple, Callable, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import ParameterGrid, cross_val_score, train_test_split
from tqdm import tqdm

def _get_xgboost():
    from xgboost import XGBClassifier
    return XGBClassifier

def _get_random_forest():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier

def _get_catboost():
    from catboost import CatBoostClassifier
    return CatBoostClassifier

def split_data(
    df: pd.DataFrame,
    target_col: str,
    train_size: float = 0.7,
    val_test_split: float = 0.5,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split a DataFrame into stratified train/val/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_size), random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_split, random_state=random_state, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# Central registry mapping configuration keys to model classes
MODEL_REGISTRY: Dict[str, Callable[[], type[BaseEstimator]]] = {
    "xgboost": _get_xgboost,
    "random_forest": _get_random_forest,
    "catboost": _get_catboost,
}

def train_model_with_gridsearch(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: Dict[str, list],
    cv: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Perform grid search CV for the selected model from MODEL_REGISTRY."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_type]()
    grid = list(ParameterGrid(param_grid))
    
    # Identify categorical column names for CatBoost
    cat_features = None
    if model_type == "catboost":
        cat_features = list(X_train.select_dtypes(include=["object", "category"]).columns)

    best_score = -1.0
    best_params = None
    best_estimator = None

    print(f"Running GridSearch for {model_type} across {len(grid)} hyperparameter combinations...")
    for params in tqdm(grid):
        if model_type == "catboost":
            estimator = model_cls(**params, random_state=random_state, verbose=0)
            fit_params = {"cat_features": cat_features} if cat_features else None
            # Pass fit_params inside params parameter for sklearn's cross_val_score
            scores = cross_val_score(estimator, X_train, y_train, cv=cv, params=fit_params)
        elif model_type == "xgboost":
            estimator = model_cls(**params, random_state=random_state, enable_categorical=True)
            scores = cross_val_score(estimator, X_train, y_train, cv=cv)
        else:
            estimator = model_cls(**params, random_state=random_state)
            scores = cross_val_score(estimator, X_train, y_train, cv=cv)

        mean_score = float(np.mean(scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    # Refit best estimator on full training set
    if model_type == "catboost":
        best_estimator = model_cls(**best_params, random_state=random_state, verbose=0)
        if cat_features:
            best_estimator.fit(X_train, y_train, cat_features=cat_features)
        else:
            best_estimator.fit(X_train, y_train)
    elif model_type == "xgboost":
        best_estimator = model_cls(**best_params, random_state=random_state, enable_categorical=True)
        best_estimator.fit(X_train, y_train)
    else:
        best_estimator = model_cls(**best_params, random_state=random_state)
        best_estimator.fit(X_train, y_train)

    return {
        "best_estimator_": best_estimator,
        "best_params_": best_params,
        "best_score_": best_score,
    }

