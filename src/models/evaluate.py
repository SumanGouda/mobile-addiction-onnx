"""Model evaluation utilities."""
from typing import Any, Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, roc_auc_score)


def evaluate_classifier(
    model: Any,
    X: Union[pd.DataFrame, np.ndarray],
    y: Union[pd.Series, np.ndarray],
    verbose: bool = True,
) -> Dict[str, Any]:
    """Evaluate a trained classifier and return standard metrics.

    Args:
        model: A fitted classifier with `.predict()` and optionally `.predict_proba()`.
        X: Feature matrix to predict on.
        y: True binary or multi-class labels.
        verbose: If True, prints formatted metrics to stdout.

    Returns:
        A dictionary containing accuracy, roc_auc (if available), classification report
        (as a nested dict and string), and confusion matrix as a JSON-serializable list.
    """
    preds = model.predict(X)
    acc = float(accuracy_score(y, preds))
    cm_array = confusion_matrix(y, preds)
    cm_list = cm_array.tolist()  # Converted to list for clean JSON/MLflow serialization

    # Compute ROC-AUC if probability estimates are available
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X)
            if probs.ndim == 2 and probs.shape[1] == 2:
                roc_auc = float(roc_auc_score(y, probs[:, 1]))
            elif probs.ndim == 2 and probs.shape[1] > 2:
                roc_auc = float(roc_auc_score(y, probs, multi_class="ovr"))
        except Exception:
            roc_auc = None

    # Generate both dictionary and formatted string reports
    report_dict = classification_report(y, preds, output_dict=True)
    report_str = classification_report(y, preds)

    if verbose:
        print(f"Accuracy: {acc:.4f}")
        if roc_auc is not None:
            print(f"ROC-AUC: {roc_auc:.4f}")
        print("\nClassification Report:\n", report_str)
        print("Confusion Matrix:\n", cm_array)

    metrics = {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "report_dict": report_dict,
        "report": report_str,
        "confusion_matrix": cm_list,
    }

    return metrics
