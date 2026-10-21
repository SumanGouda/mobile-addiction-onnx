from pathlib import Path
import os

# ==========================================
# System & Path Configurations
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==========================================
# Data Processing Paths
# ==========================================
RAW_DATA_TEST = PROJECT_ROOT / "data" / "interim" / "test.csv" 
RAW_DATA_TRAIN = PROJECT_ROOT / "data" / "interim" / "train.csv" 

PROCESSED_DATA_TEST = PROJECT_ROOT / "data" / "processed" / "test.csv" 
PROCESSED_DATA_TRAIN = PROJECT_ROOT / "data" / "processed" / "train.csv" 

METADATA = PROJECT_ROOT / "data" / "processed" / "metadata"

# ==========================================
# Preprocessing Rules & Maps
# ==========================================
ENCODING_MAP = {
    "ohe_encoding": ["gender", "stress_level", "academic_work_impact"],
}

# ==========================================
# Model Training Settings
# ==========================================
EXPERIMENT_NAME = "Mobile_Addiction_Classification_1"
from xgboost import XGBClassifier
MODEL = XGBClassifier(random_state=42, eval_metric="logloss")
PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.2],
    "subsample": [0.8, 1.0],
}
TARGET_FEAT = "addicted_label"
DATASET = PROCESSED_DATA_TRAIN

# ==========================================
# Model Serving Settings
# ==========================================
MODEL_TYPE = os.getenv("MODEL_TYPE", "xgboost")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")
MODEL_NAME = f"MobileAddiction_{MODEL_TYPE}"