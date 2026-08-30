"""Inference logic for the prediction API — loads a specific trained run
and applies the same preprocessing used during training/batch inference."""
import pandas as pd
from pathlib import Path

from src.features.encoding import apply_gender_encoding, label_encode_columns
from src.models.predict import align_features, load_model, predict_proba
from utils.helpers import load_json_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # app/api/inference.py -> project root
CONFIGS_DIR = PROJECT_ROOT / "configs"
RUN_ID = "20260830_100720"                              # <-- set to the run you want to serve
RUN_DIR = PROJECT_ROOT / "models" / "runs" / RUN_ID


class PredictionService:
    """Loads a trained model + its encoding metadata once, and serves
    predictions on individual raw feature dicts."""

    def __init__(self):
        self.run_id = RUN_ID
        self.model = load_model(str(RUN_DIR / "xgb_addiction_model.pkl"))
        self.expected_features = self.model.get_booster().feature_names

        self.label_encodings = load_json_config(str(CONFIGS_DIR / "label_encodings.json"))
        self.encoding_metadata_path = str(RUN_DIR / "encoding_metadata.json")

    def predict(self, raw_input: dict) -> dict:
        """Run the full preprocessing + prediction pipeline on a single
        raw feature dict.

        Args:
            raw_input: Dict of raw feature values, matching PredictionRequest.

        Returns:
            Dict with addicted_probability, addicted_label, model_run_id.
        """
        df = pd.DataFrame([raw_input])

        df = label_encode_columns(df, self.label_encodings)
        df = apply_gender_encoding(df, metadata_path=self.encoding_metadata_path)

        df = align_features(df, self.expected_features)

        proba = predict_proba(self.model, df)[0]
        label = int(proba > 0.5)

        return {
            "addicted_probability": float(proba),
            "addicted_label": label,
            "model_run_id": self.run_id,
        }

    