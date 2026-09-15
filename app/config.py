"""Configuración del microservicio de IA.

Carga valores desde variables de entorno y un archivo `.env` (via pydantic-settings).
Coherente con `contracts/ia-api.yaml`.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros de configuración del servicio IA."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "MedCare IA — API Microservicio IA"
    app_version: str = "1.1.1"
    app_env: str = "development"

    api_prefix: str = ""

    # Infraestructura del modelo
    model_path: str = "models/model.joblib"
    model_min_version: str = "0.1.0"

    # Dataset Kaggle (C0-3 / C1-1)
    dataset_path: str = "data/KaggleV2-May-2016.csv"

    # Integración con la API Core .NET (Sprint 2+)
    core_api_url: str | None = None
    core_token: str | None = None

    @field_validator("api_prefix")
    @classmethod
    def _normalize_prefix(cls, value: str) -> str:
        value = (value or "").strip()
        if value and not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/")

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración (cacheada)."""
    return Settings()
