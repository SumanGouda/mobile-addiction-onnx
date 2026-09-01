"""Inference logic for the prediction API — loads a trained MLflow run
and applies the same preprocessing used during training/batch inference."""
import os
from pathlib import Path

import pandas as pd
import xgboost as xgb

from src.features.encoding import apply_gender_encoding, label_encode_columns
from src.models.predict import align_features, predict_proba
from utils.helpers import load_json_config

# Project Root Resolution (app/api/inference.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"
MLRUNS_DIR = PROJECT_ROOT / "mlruns" / "1"

RUN_ID = os.getenv("MLFLOW_RUN_ID", "8a13517b9bec41a78485b45d02345200")


class PredictionService:
    """Loads a trained model + its encoding metadata once from MLflow,
    and serves predictions on individual raw feature dicts."""

    def __init__(self, run_id: str = None):
        self.run_id = run_id or RUN_ID
 
        models_dir = MLRUNS_DIR / "models"
        subfolders = list(models_dir.glob("m-*"))

        if subfolders:
            model_file = subfolders[0] / "artifacts" / "model.ubj"
            if not model_file.exists(): 
                model_file = subfolders[0] / "model.ubj"
        else:
            model_file = MLRUNS_DIR / self.run_id / "artifacts" / "model" / "model.ubj"

        if not model_file.exists():
            raise FileNotFoundError(f"Could not locate model file at: {model_file}")
 
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(model_file))
 
        self.expected_features = self.model.get_booster().feature_names
 
        self.label_encodings = load_json_config(str(CONFIGS_DIR / "label_encodings.json"))

        encoding_file = MLRUNS_DIR / self.run_id / "artifacts" / "encoding" / "encoding_metadata_temp.json"
        if not encoding_file.exists():  # Fallback if artifact folder is omitted
            encoding_file = MLRUNS_DIR / self.run_id / "encoding" / "encoding_metadata_temp.json"

        self.encoding_metadata_path = str(encoding_file)

    def predict(self, raw_input: dict) -> dict:
        """Run the full preprocessing + prediction pipeline on a single raw feature dict."""
        df = pd.DataFrame([raw_input])

        df = label_encode_columns(df, self.label_encodings)
        df = apply_gender_encoding(df, metadata_path=self.encoding_metadata_path)
        df = align_features(df, self.expected_features)

        proba = float(predict_proba(self.model, df)[0])
        label = int(proba > 0.5)

        return {
            "addicted_probability": proba,
            "addicted_label": label,
            "model_run_id": self.run_id,
        }
    