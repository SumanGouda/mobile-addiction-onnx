"""End-to-end training pipeline: load → clean → encode → train → evaluate.
All run tracking (params, metrics, artifacts, model) is handled by MLflow
instead of manual JSON files/run folders.
"""
import json
from pathlib import Path

import mlflow
import mlflow.xgboost
import pandas as pd

from src.data.cleaning import fill_missing_by_correlation, fill_missing_with_median
from src.data.loader import load_raw_data
from src.features.encoding import label_encode_columns, one_hot_encode_gender
from src.models.evaluate import evaluate_classifier
from src.models.train import split_data, train_xgb_with_gridsearch
from utils.helpers import ensure_path, load_json_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"

mlflow.set_experiment("mobile-addiction-prediction")


def main():
    cfg = load_json_config(CONFIGS_DIR / "pipeline_config.json")
    label_encodings = load_json_config(CONFIGS_DIR / "label_encodings.json")
    param_grid = load_json_config(CONFIGS_DIR / f"{cfg['model_type']}_param_grid.json")

    with mlflow.start_run(run_name=cfg["model_type"]):
        # Log config as parameters so every run records exactly what
        mlflow.log_params({
            "model_type": cfg["model_type"],
            "target_col": cfg["target_col"],
            "random_state": cfg["random_state"],
            "cv_folds": cfg["cv_folds"],
            "train_size": cfg["train_size"],
            "val_test_split": cfg["val_test_split"],
        })
        mlflow.log_dict(param_grid, "param_grid.json")
        mlflow.log_dict(label_encodings, "label_encodings.json")

        raw_data_path = DATA_RAW_DIR / cfg["raw_train_filename"]
        df = load_raw_data(str(raw_data_path))

        # encoding_metadata still needs to be a real file on disk temporarily,
        # since one_hot_encode_gender writes to a path — log it to MLflow
        # right after, then it's preserved as a run artifact
        temp_metadata_path = PROCESSED_DIR / "encoding_metadata_temp.json"
        ensure_path(temp_metadata_path, is_file=True)
        df = one_hot_encode_gender(df, save_metadata_path=str(temp_metadata_path))
        mlflow.log_artifact(str(temp_metadata_path), artifact_path="encoding")

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
        mlflow.log_dict(imputation_log, "imputation_log.json")

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

        grid_search = train_xgb_with_gridsearch(
            X_train,
            y_train,
            param_grid,
            cv=cfg["cv_folds"],
            random_state=cfg["random_state"],
        )
        best_model = grid_search["best_estimator_"]

        mlflow.log_params(grid_search["best_params_"])
        mlflow.log_metric("best_cv_score", grid_search["best_score_"])

        print("Validation metrics:")
        val_results = evaluate_classifier(best_model, X_val, y_val)

        print("Test metrics:")
        test_results = evaluate_classifier(best_model, X_test, y_test)

        mlflow.log_metric("val_accuracy", val_results["accuracy"])
        mlflow.log_metric("test_accuracy", test_results["accuracy"])
        mlflow.log_text(val_results["report"], "val_classification_report.txt")
        mlflow.log_text(test_results["report"], "test_classification_report.txt")
        mlflow.log_dict(
            {"confusion_matrix": test_results["confusion_matrix"].tolist()},
            "test_confusion_matrix.json",
        )

        mlflow.xgboost.log_model(best_model, "model")

        run_id = mlflow.active_run().info.run_id
        print(f"\nRun logged to MLflow. Run ID: {run_id}")
        print("View results with: mlflow ui")

if __name__ == "__main__":
    main()
