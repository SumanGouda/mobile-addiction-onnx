import json
from pathlib import Path
from typing import Any, List, Optional, Union

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient


def load_model_from_registry(
    model_name: str, version: str, tracking_uri: Optional[str] = None
) -> Any:
    """Load a trained model artifact directly from the MLflow Model Registry.

    Args:
        model_name: Name of the registered model in MLflow.
        version: Version string or number in the registry.
        tracking_uri: Optional MLflow tracking URI override.

    Returns:
        The loaded MLflow pyfunc model object and associated run_id.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()
    model_ver_info = client.get_model_version(name=model_name, version=version)
    run_id = model_ver_info.run_id

    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pyfunc.load_model(model_uri)

    return model, run_id


def load_model(path: Union[str, Path]) -> Any:
    """Load a trained model artifact from disk using joblib.

    Args:
        path: Path to the serialized model file (.pkl, .joblib).

    Returns:
        The loaded model object.
    """
    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at path: {model_path.resolve()}"
        )

    if not model_path.is_file():
        raise ValueError(
            f"Specified path is not a file: {model_path.resolve()}"
        )

    if model_path.stat().st_size == 0:
        raise ValueError(f"Model file is empty: {model_path.resolve()}")

    try:
        model = joblib.load(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model from {model_path.resolve()}: {str(e)}"
        ) from e

    return model


def align_features(
    df: pd.DataFrame,
    expected_features: List[str],
    fill_value: Union[int, float] = 0,
) -> pd.DataFrame:
    """Align a DataFrame's columns to match the exact features a model was
    trained on, filling any missing columns with 0 and ordering them correctly.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected df to be a pandas DataFrame, got {type(df).__name__}"
        )

    if not isinstance(expected_features, (list, tuple)):
        raise TypeError(
            f"Expected expected_features to be a list/tuple, got {type(expected_features).__name__}"
        )

    if not expected_features:
        raise ValueError("expected_features list cannot be empty.")

    aligned_df = df.reindex(columns=expected_features, fill_value=fill_value)

    return aligned_df


def predict_score(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Generate class-1 probabilities from a fitted classifier or MLflow PyFunc model."""
    target_model = getattr(model, "_model_impl", model)
    if hasattr(target_model, "python_model"):
        target_model = getattr(target_model, "python_model", target_model)

    if hasattr(target_model, "predict_proba"):
        probabilities = target_model.predict_proba(X)

        if isinstance(probabilities, pd.DataFrame):
            probabilities = probabilities.values

        probabilities = np.asarray(probabilities)

        if probabilities.ndim == 2 and probabilities.shape[1] > 1:
            return probabilities[:, 1]
        return probabilities.ravel()

    if hasattr(model, "predict"):
        preds = model.predict(X)

        if isinstance(preds, pd.DataFrame):
            preds = preds.values

        preds = np.asarray(preds)

        if preds.ndim == 2 and preds.shape[1] > 1:
            return preds[:, 1]

        return preds.ravel()

    raise AttributeError(
        "The provided model does not support probability output via 'predict_proba' or 'predict'."
    )
