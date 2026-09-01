"""Inference utilities."""
import joblib
import numpy as np
import pandas as pd


def load_model(path: str):
    """Load a trained model from disk.

    Args:
        path: Path to the saved model file (e.g. .pkl).

    Returns:
        The loaded model object.
    """
    return joblib.load(path)


def align_features(df: pd.DataFrame, expected_features: list) -> pd.DataFrame:
    """Align a DataFrame's columns to match the exact features a model was
    trained on, filling any missing columns with 0.

    Args:
        df: The DataFrame to align.
        expected_features: The ordered list of feature names the model
            expects (e.g. from `model.get_booster().feature_names`).

    Returns:
        A DataFrame with exactly the expected columns, in the expected order.
    """
    return df.reindex(columns=expected_features, fill_value=0)


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Generate class-1 probabilities from a fitted classifier or MLflow PyFunc model.

    Args:
        model: Fitted classifier (with `.predict_proba()`) or an MLflow `PyFuncModel`.
        X: Feature matrix to predict on.

    Returns:
        Array of predicted probabilities for the positive class.
    """ 
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()
 
    if hasattr(model, "predict"):
        preds = model.predict(X)
        if isinstance(preds, pd.DataFrame):
            preds = preds.values
        elif not isinstance(preds, np.ndarray):
            preds = np.array(preds)
 
        if preds.ndim == 2 and preds.shape[1] > 1:
            return preds[:, 1]
         
        return preds.ravel()

    raise AttributeError("The provided model has neither 'predict_proba' nor 'predict' methods.")
