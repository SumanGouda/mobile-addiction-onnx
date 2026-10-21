from typing import Literal 
from pydantic import BaseModel, ConfigDict, Field

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
    gender: Literal["Male", "Female", "Other"]
    stress_level: Literal["Low", "Medium", "High"]
    academic_work_impact: Literal["Yes", "No"]


class PredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    addicted_probability: float = Field(..., ge=0.0, le=1.0)    
    addicted_label: int = Field(..., ge=0, le=1)
    model_name: str
    model_version: str
    model_run_id: str
