"""Pydantic request/response schemas for the prediction API."""
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Raw input features for a single prediction. All fields required —
    no defaults, so Pydantic rejects requests with any field missing."""

    age: float
    daily_screen_time_hours: float
    social_media_hours: float
    gaming_hours: float
    work_study_hours: float
    sleep_hours: float
    notifications_per_day: float
    app_opens_per_day: float
    weekend_screen_time: float
    gender: str = Field(..., description="One of: Male, Female, Other")
    stress_level: str = Field(..., description="One of: Low, Medium, High")
    academic_work_impact: str = Field(..., description="Yes or No")


class PredictionResponse(BaseModel):
    """Prediction output."""

    addicted_probability: float
    addicted_label: int
    model_run_id: str
    