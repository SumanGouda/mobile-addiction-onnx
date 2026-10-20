"""Inference logic for the prediction API — loads a registered model from
MLflow's Model Registry using dynamic Pydantic settings."""
import json
from pathlib import Path

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

from app.core.config import settings
from src.features.encoding import apply_gender_encoding, label_encode_columns
from src.models.predict import align_features
from utils.helpers import load_json_config

# Set MLflow tracking URI directly from settings
mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)


class PredictionService:

    def __init__(
        self, model_name: str = None, version: str = None
    ):
        self.model_name = model_name or settings.MODEL_NAME
        self.version = version or settings.MODEL_VERSION

        # 1. Resolve run_id from Model Registry
        client = MlflowClient()
        model_ver_info = client.get_model_version(
            name=self.model_name, version=self.version
        )
        self.run_id = model_ver_info.run_id

        # 2. Load model via pyfunc (and keep underlying model reference if available)
        self.model_uri = f"runs:/{self.run_id}/model"
        self.pyfunc_model = mlflow.pyfunc.load_model(self.model_uri)

        # Extract native XGBoost model to access predict_proba / DMatrix predictions
        try:
            if hasattr(self.pyfunc_model._model_impl, "xgboost_model"):
                self.native_model = self.pyfunc_model._model_impl.xgboost_model
            else:
                self.native_model = mlflow.xgboost.load_model(self.model_uri)
        except Exception:
            self.native_model = None

        # 3. Extract expected feature names
        try:
            booster = (
                self.native_model.get_booster()
                if hasattr(self.native_model, "get_booster")
                else self.native_model
            )
            self.expected_features = booster.feature_names
        except Exception:
            self.expected_features = None

        # 4. Load label encodings
        try:
            encoding_path = download_artifacts(
                artifact_uri=f"runs:/{self.run_id}/configs/label_encodings.json"
            )
            with open(encoding_path, "r") as f:
                self.label_encodings = json.load(f)
        except Exception:
            self.label_encodings = load_json_config(
                str(settings.CONFIGS_DIR / "label_encodings.json")
            )

        # 5. Download gender encoding metadata
        try:
            self.encoding_metadata_path = download_artifacts(
                artifact_uri=f"runs:/{self.run_id}/encoding/encoding_metadata_temp.json"
            )
        except Exception:
            self.encoding_metadata_path = str(
                settings.CONFIGS_DIR / "encoding_metadata_temp.json"
            )

    def predict(self, raw_input: dict) -> dict:
        """Run full preprocessing + prediction pipeline on a single raw feature dict."""
        df = pd.DataFrame([raw_input])

        df = label_encode_columns(df, self.label_encodings)
        df = apply_gender_encoding(
            df, metadata_path=self.encoding_metadata_path
        )

        if self.expected_features:
            df = align_features(df, self.expected_features)

        # Get continuous probability score
        if self.native_model and hasattr(self.native_model, "predict_proba"):
            # Native XGBClassifier: predict_proba returns [[prob_0, prob_1]]
            proba_array = self.native_model.predict_proba(df)
            proba = float(proba_array[0][1])
        elif self.native_model and hasattr(self.native_model, "predict"):
            import xgboost as xgb

            dmatrix = xgb.DMatrix(df)
            proba_array = self.native_model.predict(dmatrix)
            proba = float(proba_array[0])
        else:
            # Fallback to pyfunc prediction
            pred_output = self.pyfunc_model.predict(df)
            proba = float(
                pred_output[0][1]
                if getattr(pred_output, "ndim", 1) > 1
                else pred_output[0]
            )

        label = int(proba > 0.5)

        return {
            "addicted_probability": proba,
            "addicted_label": label,
            "model_name": self.model_name,
            "model_version": self.version,
            "model_run_id": self.run_id,
        }
    