"""End-to-end training pipeline: load → clean → encode → train → evaluate → save.
All artifacts from a single run (model, metadata, logs, metrics) are saved
together under models/runs/<run_id>/.
"""
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

import joblib

from src.data.cleaning import fill_missing_by_correlation, fill_missing_with_median
from src.data.loader import load_raw_data
from src.features.encoding import label_encode_columns, one_hot_encode_gender
from src.models.evaluate import evaluate_classifier
from src.models.train import split_data, train_xgb_with_gridsearch

from utils.helpers import ensure_path, load_json_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # pipeline/train_pipeline.py -> project root
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "models" / "runs"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def main():
    cfg = load_json_config(CONFIGS_DIR / "pipeline_config.json")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / run_id
    ensure_path(run_dir, is_file=False)

    for cfg_file in ["pipeline_config.json", f"{cfg['model_type']}_param_grid.json", "label_encodings.json"]:
        shutil.copy(CONFIGS_DIR / cfg_file, run_dir / cfg_file)

    raw_data_path = DATA_RAW_DIR / cfg["raw_train_filename"]
    df = load_raw_data(str(raw_data_path))

    enc_metadata_path = run_dir / cfg["encoding_metadata_filename"]
    ensure_path(enc_metadata_path, is_file=True)
    df = one_hot_encode_gender(df, save_metadata_path=str(enc_metadata_path))

    label_encodings = load_json_config(CONFIGS_DIR / "label_encodings.json")
    df = label_encode_columns(df, label_encodings)

    imputation_log = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df, msg = fill_missing_by_correlation(
                df,
                col,
                threshold=cfg["imputation_threshold"],
                missing_threshold=cfg["imputation_missing_threshold"],
                n_related=cfg["imputation_n_related"],
            )
            imputation_log[col] = msg
    df = fill_missing_with_median(df)

    imputed_log_path = run_dir / cfg["imputation_log_filename"]
    ensure_path(imputed_log_path, is_file=True)
    with open(imputed_log_path, "w") as f:
        json.dump(imputation_log, f, indent=2)

    prc_csv_path = PROCESSED_DIR / cfg["processed_filename"]
    ensure_path(prc_csv_path, is_file=True)
    df.to_csv(str(prc_csv_path), index=False)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_col=cfg["target_col"],
        train_size=cfg["train_size"],
        val_test_split=cfg["val_test_split"],
        random_state=cfg["random_state"],
    )

    param_grid = load_json_config(CONFIGS_DIR / f"{cfg['model_type']}_param_grid.json")

    training_start = time.time()
    grid_search = train_xgb_with_gridsearch(
        X_train,
        y_train,
        param_grid,
        cv=cfg["cv_folds"],
        random_state=cfg["random_state"],
    )
    best_model = grid_search["best_estimator_"]
    training_duration_seconds = time.time() - training_start

    params_path = run_dir / cfg["params_filename"]
    ensure_path(params_path, is_file=True)
    with open(params_path, "w") as f:
        json.dump(grid_search["best_params_"], f, indent=2)

    print("Validation metrics:")
    val_results = evaluate_classifier(best_model, X_val, y_val)

    print("Test metrics:")
    test_results = evaluate_classifier(best_model, X_test, y_test)

    metrics = {
        "val_accuracy": val_results["accuracy"],
        "test_accuracy": test_results["accuracy"],
        "best_cv_score": grid_search["best_score_"],
        "training_duration_seconds": round(training_duration_seconds, 2),
        "val_classification_report": val_results["report"],
        "test_classification_report": test_results["report"],
        "test_confusion_matrix": test_results["confusion_matrix"].tolist(),
    }
    metrics_path = run_dir / cfg["metrics_filename"]
    ensure_path(metrics_path, is_file=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    model_path = run_dir / cfg["model_filename"]
    ensure_path(model_path, is_file=True)
    joblib.dump(best_model, str(model_path))
    print(f"Model saved to {model_path}")

    print(f"\nAll run artifacts saved under: {run_dir}")


if __name__ == "__main__":
    main()
