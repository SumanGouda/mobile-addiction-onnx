"""End-to-end training pipeline: load → clean → encode → train → evaluate.
All run tracking (params, metrics, artifacts, model) is handled by MLflow.
"""
import os
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from src.data.cleaning import fill_missing_by_correlation, fill_missing_with_median
from src.data.loader import load_raw_data
from src.features.encoding import label_encode_columns, one_hot_encode_gender
from src.models.evaluate import evaluate_classifier
from src.models.train import split_data, train_model_with_gridsearch
from utils.helpers import ensure_path, load_json_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CONFIGS_DIR = PROJECT_ROOT / "configs"
NATIVE_CATEGORICAL_MODELS = {"catboost", "xgboost", "lightgbm"}

db_path = PROJECT_ROOT / "mlflow.db"
ensure_path(db_path, is_file=True)
mlflow.set_tracking_uri(f"sqlite:///{db_path}")
mlflow.set_experiment("mobile-addiction-prediction")


def main(label_encoding: bool = True):
    cfg_pipeline = load_json_config(CONFIGS_DIR / "pipeline_config.json")
    param_grid = load_json_config(
        CONFIGS_DIR / f"{cfg_pipeline['model_type']}_param_grid.json"
    )

    # Validate configuration before initiating run
    if not label_encoding and cfg_pipeline["model_type"] not in NATIVE_CATEGORICAL_MODELS:
        raise ValueError(
            f"Model '{cfg_pipeline['model_type']}' cannot run with label_encoding=False. "
            f"Set label_encoding=True or use one of: {NATIVE_CATEGORICAL_MODELS}"
        )

    with mlflow.start_run(run_name=cfg_pipeline["model_type"]):
        # 1. Log basic configuration parameters and artifacts
        mlflow.log_params(
            {
                "model_type": cfg_pipeline["model_type"],
                "target_col": cfg_pipeline["target_col"],
                "random_state": cfg_pipeline["random_state"],
                "cv_folds": cfg_pipeline["cv_folds"],
                "train_size": cfg_pipeline["train_size"],
                "val_test_split": cfg_pipeline["val_test_split"],
                "label_encoding_enabled": label_encoding,
            }
        )

        mlflow.log_dict(param_grid, artifact_file="configs/param_grid.json")

        # 2. Data Loading & Preprocessing
        raw_data_path = DATA_RAW_DIR / cfg_pipeline["raw_train_filename"]
        df = load_raw_data(str(raw_data_path))

        temp_metadata_path = PROCESSED_DIR / "encoding_metadata_temp.json"
        
        try:
            if label_encoding:
                label_encodings = load_json_config(CONFIGS_DIR / "label_encodings.json")
                mlflow.log_dict(label_encodings, artifact_file="configs/label_encodings.json")

                ensure_path(temp_metadata_path, is_file=True)
                df = one_hot_encode_gender(df, save_metadata_path=str(temp_metadata_path))

                # Store encoding metadata in dedicated 'encoding/' subfolder
                mlflow.log_artifact(str(temp_metadata_path), artifact_path="encoding")

                df = label_encode_columns(df, label_encodings)
            else: 
                model_type = cfg_pipeline["model_type"]
                cat_cols = df.select_dtypes(include=["object", "category"]).columns

                for col in cat_cols:
                    if model_type == "xgboost":
                        df[col] = df[col].fillna("Missing").astype("category")
                    elif model_type == "catboost":
                        df[col] = df[col].fillna("Missing").astype(str)
        finally:
            if temp_metadata_path.exists():
                os.remove(temp_metadata_path)

        imputation_log = {}
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                df, msg = fill_missing_by_correlation(
                    df,
                    col,
                    threshold=cfg_pipeline["imputation_threshold"],
                    missing_threshold=cfg_pipeline["imputation_missing_threshold"],
                    n_related=cfg_pipeline["imputation_n_related"],
                )
                imputation_log[col] = msg

        df = fill_missing_with_median(df)
        mlflow.log_dict(imputation_log, artifact_file="logs/imputation_log.json")

        prc_csv_path = PROCESSED_DIR / cfg_pipeline["processed_filename"]
        ensure_path(prc_csv_path, is_file=True)
        df.to_csv(str(prc_csv_path), index=False)

        # 3. Data Splitting & Model Training
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df,
            target_col=cfg_pipeline["target_col"],
            train_size=cfg_pipeline["train_size"],
            val_test_split=cfg_pipeline["val_test_split"],
            random_state=cfg_pipeline["random_state"],
        )

        grid_search = train_model_with_gridsearch(
            model_type=cfg_pipeline["model_type"],
            X_train=X_train,
            y_train=y_train,
            param_grid=param_grid,
            cv=cfg_pipeline["cv_folds"],
            random_state=cfg_pipeline["random_state"],
        )
        best_model = grid_search["best_estimator_"]

        # Log winning hyperparameter outputs & best CV score
        mlflow.log_params(grid_search["best_params_"])
        mlflow.log_metric("best_cv_score", grid_search["best_score_"])

        # 4. Evaluation
        print("\n--- Validation Metrics ---")
        val_results = evaluate_classifier(best_model, X_val, y_val)

        print("\n--- Test Metrics ---")
        test_results = evaluate_classifier(best_model, X_test, y_test)

        # Log evaluation metrics
        mlflow.log_metric("val_accuracy", val_results["accuracy"])
        mlflow.log_metric("test_accuracy", test_results["accuracy"])

        if val_results.get("roc_auc") is not None:
            mlflow.log_metric("val_roc_auc", val_results["roc_auc"])
        if test_results.get("roc_auc") is not None:
            mlflow.log_metric("test_roc_auc", test_results["roc_auc"])

        mlflow.log_text(
            val_results["report"], artifact_file="reports/val_classification_report.txt"
        )
        mlflow.log_text(
            test_results["report"], artifact_file="reports/test_classification_report.txt"
        ) 
        cm = test_results["confusion_matrix"]
        cm_list = cm.tolist() if hasattr(cm, "tolist") else cm

        mlflow.log_dict(
            {"confusion_matrix": cm_list},
            artifact_file="reports/test_confusion_matrix.json",
        ) 
        
        # 5. Model Serialization Logging & Registration
        model_type_name = cfg_pipeline["model_type"]
        registered_name = f"MobileAddiction_{model_type_name}"

        version_description = (
            f"Model Type: {model_type_name}\n" 
            f"Label Encoding: {label_encoding}\n"
            f"Validation Accuracy: {val_results['accuracy']:.4f}\n"
            f"Test Accuracy: {test_results['accuracy']:.4f}"
        )

        # Infer input signature and input example for MLflow model artifact schema
        signature = infer_signature(X_train, best_model.predict(X_train))
        input_example = X_train.iloc[:2]

        if model_type_name == "xgboost":
            from mlflow import xgboost as mlflow_xgboost

            model_info = mlflow_xgboost.log_model(
                best_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_name,
            )
        elif model_type_name == "catboost":
            from mlflow import catboost as mlflow_catboost

            model_info = mlflow_catboost.log_model(
                best_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_name,
            )
        else:
            from mlflow import sklearn as mlflow_sklearn

            model_info = mlflow_sklearn.log_model(
                best_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                registered_model_name=registered_name,
            )

        # Update description for registered model version
        client = MlflowClient()
        client.update_model_version(
            name=registered_name,
            version=model_info.registered_model_version,
            description=version_description,
        )


if __name__ == "__main__":
    main(label_encoding=True)
    