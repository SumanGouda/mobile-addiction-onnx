"""FastAPI app serving the mobile addiction prediction model.""" 
import os
import mlflow
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, status
from configs.settings import METADATA, MODEL_NAME, MODEL_TYPE, MODEL_VERSION

from configs.settings import METADATA
from utils.helpers import get_model_feature_order
from utils.preprocessing import preprocess_single_row
from app.api.schemas import PredictionRequest, PredictionResponse
from utils.helpers import get_model_feature_order, load_json_config, load_registered_model

logger = logging.getLogger("prediction_api")
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # app/api/main.py -> project root 

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  
mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())  


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Any failure here stops the server from starting, which is intended.
    model = load_registered_model(MODEL_TYPE, f"models:/{MODEL_NAME}/{MODEL_VERSION}")
 
    app.state.model          = model
    app.state.feature_order  = get_model_feature_order(model)
    app.state.encoding_meta  = load_json_config(Path(METADATA) / "encoding_metadata_train.json")
    app.state.fill_meta      = load_json_config(Path(METADATA) / "fill_missing_train.json")
    app.state.run_id         = MlflowClient().get_model_version(MODEL_NAME, MODEL_VERSION).run_id
    yield
 
app = FastAPI(title="Mobile Addiction Prediction API", lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "Mobile Addiction Prediction API is running. Visit /docs for documentation."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_run_id": app.state.run_id,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    state = app.state
    try:
        row = preprocess_single_row(request.model_dump(), state.encoding_meta, state.fill_meta)
        row = row[state.feature_order]  # same columns, same order as training
        proba = float(state.model.predict_proba(row)[0, 1])
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed",
        )

    return {
        "addicted_probability": proba,
        "addicted_label": int(proba > 0.5),
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_run_id": state.run_id,
    }
