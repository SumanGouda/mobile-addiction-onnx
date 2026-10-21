import json
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns 
from typing import Any
from pathlib import Path 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def load_registered_model(model_type: str, uri: str):
    """Load a model with robust flavor fallback for sklearn-logged models."""
    if model_type == "xgboost":
        try:
            import mlflow.xgboost
            return mlflow.xgboost.load_model(uri)
        except Exception:
            import mlflow.sklearn
            return mlflow.sklearn.load_model(uri)
            
    if model_type == "catboost":
        import mlflow.catboost
        return mlflow.catboost.load_model(uri)
        
    import mlflow.sklearn
    return mlflow.sklearn.load_model(uri)


def load_json_config(path) -> dict:
    """Load a JSON file, failing with a clear error if it's missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def check_file_exists(path: Path, expected_type: str, create_if_missing: bool = True,) -> bool:
    """Check whether a file exists and is of the expected type, optionally
    creating an empty one (plus its parent directories) if it doesn't.

    Args:
        path: The file path to check.
        expected_type: Expected file extension, e.g. ".csv", ".json".
            Matching is case-insensitive and the leading dot is optional
            (both "csv" and ".csv" work).
        create_if_missing: If True, creates parent dirs + an empty file
            when missing.

    Returns:
        True if the file already existed, False if it was missing
        (and possibly just created).

    Raises:
        ValueError: If the file's extension does not match expected_type.
    """
    path = Path(path)

    # Normalize: strip leading dot, lowercase (so ".csv" / "csv" / ".CSV" all work)
    expected = expected_type.lower().lstrip(".")
    actual = path.suffix.lower().lstrip(".")

    if actual != expected:
        raise ValueError(
            f"File type mismatch for '{path}': expected '.{expected}', "
            f"got '{path.suffix or '<no extension>'}'."
        )

    existed = path.exists()
    if not existed and create_if_missing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return existed


def apply_ohe_with_reference(df: pd.DataFrame, column: str, reference_meta: dict) -> tuple[pd.DataFrame, dict]:
    """Apply a previously fitted one-hot encoding to new data, using the
    exact columns recorded during training.

    Returns:
        (dummies, metadata) - dummies match the training schema exactly;
        metadata records what was applied and any categories unseen in training.
    """
    expected_columns = reference_meta["encoded_columns"]
    known_categories = set(reference_meta["all_categories"])
    seen_categories = set(df[column].dropna().unique().tolist())

    dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
    for col in expected_columns:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[expected_columns]

    metadata = {
        "original_column": column,
        "encoded_columns": expected_columns,
        "dropped_baseline_category": reference_meta["dropped_baseline_category"],
        "categories_in_batch": sorted(seen_categories),
        "unseen_categories": sorted(seen_categories - known_categories),
    }
    return dummies, metadata


def generate_performance_plots(chunks_data):
    """Generates visual classification diagnostics (Confusion Matrices) across multiple data chunks.

    Parameters:
    -----------
    chunks_data : dict
        Nested dictionary containing ground truth and prediction arrays per
        chunk: {
            "Train": {"y_true": y_train, "y_pred": train_preds},
            "Validation": {"y_true": y_val, "y_pred": val_preds}
        }

    Returns:
    --------
    fig : matplotlib.figure.Figure
        Rendered figure containing Confusion Matrix heatmaps.
    """
    num_chunks = len(chunks_data)
    fig, axes = plt.subplots(
        nrows=1,
        ncols=num_chunks,
        figsize=(6 * num_chunks, 5),
        tight_layout=True,
    )
    # Ensure axes handle single-chunk cases safely
    if num_chunks == 1:
        axes = np.array([axes])

    for idx, (chunk_name, data) in enumerate(chunks_data.items()):
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        
        cm = confusion_matrix(y_true, y_pred)
        ax = axes[idx] if num_chunks > 1 else axes[0]

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            cbar=False
        )
        ax.set_title(f"{chunk_name}: Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    return fig


def get_model_feature_order(model: Any) -> list[str]:
    """
        Return the feature column names a trained model expects, in training order.

        Works with sklearn-style estimators (including XGBClassifier), CatBoost,
        a raw XGBoost Booster, and MLflow pyfunc models that have a logged signature.
    """ 
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    # CatBoost
    if getattr(model, "feature_names_", None):
        return list(model.feature_names_)

    # XGBoost sklearn wrapper or raw Booster
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    if getattr(booster, "feature_names", None):
        return list(booster.feature_names)

    if getattr(model, "feature_name_", None):
        return list(model.feature_name_)

    # MLflow pyfunc model (uses the signature logged at training time)
    if hasattr(model, "metadata") and model.metadata.signature is not None:
        return list(model.metadata.get_input_schema().input_names())

    raise ValueError(
        "Could not read feature names from this model. It was probably trained "
        "on a numpy array instead of a DataFrame, so no column names were stored."
    )
