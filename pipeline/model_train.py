import mlflow
import pandas as pd
import mlflow.sklearn
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt  
from utils.helpers import check_file_exists
from utils.modeling import train_model, evaluate_model
from utils.helpers import generate_performance_plots
from sklearn.model_selection import train_test_split
from configs.settings import (
    PROCESSED_DATA_TEST, PROCESSED_DATA_TRAIN, MODEL, 
    PARAM_GRID, TARGET_FEAT, EXPERIMENT_NAME
)


def main(model: Any, dataset: str | Path, grid_params: dict, target_feat: str, exp_name: str,) -> None:
    if not check_file_exists(dataset, "csv", create_if_missing=False):
        raise FileNotFoundError(f"Training dataset not found: {dataset}") 
    
    project_root = Path(__file__).resolve().parents[1]  
    mlruns_dir = project_root / "mlruns" 

    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name=f"Predict_{target_feat}") as run:
        print(f"Active MLflow Run ID: {run.info.run_id}")
        mlflow.set_tags(
            {
                "target_feat_feature": target_feat,
                "model_type": type(model).__name__,
                "pipeline_stage": "training",
            }
        )  
        df = pd.read_csv(dataset)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[target_feat])

        print("Training model...")
        train_results = train_model(train_df, val_df, model, target_feat, grid_params,)
        best_model = train_results["model"]
        best_params = train_results["best_params"]
        if best_params:
            mlflow.log_params(best_params)

        print("Evaluating model...")
        train_eval = evaluate_model(best_model, train_df, target_feat, task_type="classification")
        val_eval = evaluate_model(best_model, val_df, target_feat, task_type="classification")
        for metric_name, val in train_eval["metrics"].items():
            mlflow.log_metric(f"train_{metric_name}", val)
        for metric_name, val in val_eval["metrics"].items():
            mlflow.log_metric(f"val_{metric_name}", val)

        print("Generating performance plots...")
        chunks_data = {
            "Train": {
                "y_true": train_eval["y_true"],
                "y_pred": train_eval["predictions"],
            },
            "Validation": {
                "y_true": val_eval["y_true"],
                "y_pred": val_eval["predictions"],
            },
        }
        fig = generate_performance_plots(chunks_data)
        mlflow.log_figure(fig, artifact_file=f"plots/{target_feat}_performance.png")
        plt.close(fig)

        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE
        )
        print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main(MODEL, PROCESSED_DATA_TRAIN, PARAM_GRID, TARGET_FEAT, EXPERIMENT_NAME)
    