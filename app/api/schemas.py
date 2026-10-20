from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float
    daily_screen_time_hours: float
    social_media_hours: float
    gaming_hours: float
    work_study_hours: float
    sleep_hours: float
    notifications_per_day: float
    app_opens_per_day: float
    weekend_screen_time: float
    gender: str
    stress_level: str
    academic_work_impact: str


class PredictionResponse(BaseModel):
    addicted_probability: float = Field(..., ge=0.0, le=1.0)    # ge=0.0 means Greater than or Equal to 0.0, and le=1.0 means Less than or Equal to 1.0
    addicted_label: int = Field(..., ge=0, le=1)
    model_name: str
    model_version: str
    model_run_id: str
