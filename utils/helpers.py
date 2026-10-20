from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
    """Generates visual evaluation diagnostics across multiple data chunks.

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
        Rendered figure containing Actual vs Predicted and Residual plots.
    """
    num_chunks = len(chunks_data)
    fig, axes = plt.subplots(
        nrows=num_chunks,
        ncols=2,
        figsize=(12, 5 * num_chunks),
        tight_layout=True,
    )
    # Ensure axes matrix handles single-chunk cases safely
    if num_chunks == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (chunk_name, data) in enumerate(chunks_data.items()):
        y_true = np.array(data["y_true"])
        y_pred = np.array(data["y_pred"])
        residuals = y_true - y_pred

        # Subplot 1: Actual vs. Predicted
        ax_act = axes[idx, 0]
        ax_act.scatter(y_true, y_pred, alpha=0.5, edgecolors="k", linewidth=0.5)

        # Perfect prediction reference line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax_act.plot(
            [min_val, max_val],
            [min_val, max_val],
            color="red",
            linestyle="--",
            label="Ideal Fit",
        )
        ax_act.set_title(f"{chunk_name}: Actual vs Predicted Price")
        ax_act.set_xlabel("Actual Price")
        ax_act.set_ylabel("Predicted Price")
        ax_act.legend()
        ax_act.grid(True, linestyle=":", alpha=0.6)

        # Subplot 2: Residual Distribution
        ax_res = axes[idx, 1]
        sns.histplot(
            residuals,
            kde=True,
            ax=ax_res,
            color="teal",
            line_kws={"linewidth": 2},
        )
        ax_res.axvline(
            x=0, color="red", linestyle="--", label="Zero Residual Line"
        )
        ax_res.set_title(f"{chunk_name}: Residual Error Distribution")
        ax_res.set_xlabel("Residual (Actual - Predicted)")
        ax_res.set_ylabel("Frequency")
        ax_res.legend()
        ax_res.grid(True, linestyle=":", alpha=0.6)

    return fig
