"""Application settings managed via Pydantic Settings."""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App Information
    PROJECT_NAME: str = "Mobile Addiction Prediction API"
    API_V1_STR: str = "/api/v1"

    # MLflow Model Settings
    MODEL_NAME: str = "MobileAddiction_xgboost"
    MODEL_VERSION: str = "1"

    # Directory Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

    @property
    def CONFIGS_DIR(self) -> Path:
        return self.PROJECT_ROOT / "configs"

    @property
    def MLFLOW_DB_PATH(self) -> Path:
        return self.PROJECT_ROOT / "mlflow.db"

    @property
    def MLFLOW_TRACKING_URI(self) -> str:
        return f"sqlite:///{self.MLFLOW_DB_PATH}"

    # Pydantic Settings Configuration
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
