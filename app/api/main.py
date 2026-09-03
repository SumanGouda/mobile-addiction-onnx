"""FastAPI app serving the mobile addiction prediction model."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

from app.api.inference import PredictionService
from app.api.schemas import PredictionRequest, PredictionResponse

prediction_service: PredictionService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global prediction_service
    # Load model and preprocessing artifacts once at application startup
    prediction_service = PredictionService()
    yield
    # Clean up resources on shutdown if needed
    prediction_service = None


app = FastAPI(title="Mobile Addiction Prediction API", lifespan=lifespan)


@app.get("/health")
def health():
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_530_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_530_SERVICE_UNAVAILABLE,
            detail="Prediction service not initialized",
        )
    return {
        "model_name": prediction_service.model_name,
        "model_version": prediction_service.version,
        "run_id": prediction_service.run_id,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if prediction_service is None:
        raise HTTPException(
            status_code=status.HTTP_530_SERVICE_UNAVAILABLE,
            detail="Prediction service unavailable",
        )

    try:
        # Pydantic v2 compatible dict extraction
        raw_input = request.model_dump()
        result = prediction_service.predict(raw_input)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution failed: {str(e)}",
        )
    