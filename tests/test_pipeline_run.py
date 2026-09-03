from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from pipeline.run_pipeline import main   


@patch("pipeline.run_pipeline.train_model_with_gridsearch")
@patch("pipeline.run_pipeline.mlflow")
def test_pipeline_dry_run(mock_mlflow, mock_gridsearch): 
    """Verifies that run_pipeline executes end-to-end without running full training."""

    # 1. Mock the heavy grid search output
    dummy_model = MagicMock()
    dummy_model.predict.side_effect = lambda X: np.zeros(len(X))
    dummy_model.predict_proba.side_effect = lambda X: np.tile([0.8, 0.2], (len(X), 1))

    mock_gridsearch.return_value = {
        "best_estimator_": dummy_model,
        "best_params_": {"max_depth": 3, "n_estimators": 10},
        "best_score_": 0.95,
    }

    # 2. Mock MLflow active run & artifact methods
    mock_run = MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.xgboost.log_model.return_value.registered_model_version = "1"

    # 3. Execute the pipeline
    try:
        main(label_encoding=True)
        execution_success = True
    except Exception as e:
        execution_success = False
        pytest.fail(f"Pipeline execution failed with error: {e}")

    assert execution_success
    assert mock_gridsearch.called
    assert mock_mlflow.log_params.called
    assert mock_mlflow.log_metric.called
