"""Model evaluation utilities."""
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_classifier(
    model, X, y, verbose: bool = True
) -> Dict[str, object]:
    """Evaluate a trained classifier and return standard metrics.

    Args:
        model: A fitted classifier with a `.predict()` method.
        X: Feature matrix to predict on.
        y: True labels.
        verbose: If True, prints the metrics as well as returning them.

    Returns:
        A dict with accuracy, classification report string, and confusion
        matrix array.
    """
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    report = classification_report(y, preds)
    cm = confusion_matrix(y, preds)

    if verbose:
        print(f"Accuracy: {acc:.4f}")
        print(report)
        print(cm)

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}
