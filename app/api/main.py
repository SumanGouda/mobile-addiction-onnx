"""FastAPI app serving the mobile addiction prediction model."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.inference import PredictionService
from app.api.schemas import PredictionRequest, PredictionResponse

prediction_service: PredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_service
    prediction_service = PredictionService()  # loaded once at startup
    yield


app = FastAPI(title="Mobile Addiction Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {"run_id": prediction_service.run_id}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    result = prediction_service.predict(request.model_dump())
    return result
