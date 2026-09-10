"""Schemas de health check (contrato ia-api.yaml → GET /health)."""

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Estado de salud del servicio."""

    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    version: str
