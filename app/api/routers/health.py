"""Health check del servicio (GET /health, sin autenticación)."""

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas.health import HealthResponse
from app.services.model_registry import get_model_registry

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Health check del servicio IA")
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Estado del servicio: salud, modelo cargado y versión del contrato."""
    registry = get_model_registry()
    model_loaded = registry.is_loaded
    status = "healthy" if model_loaded else "degraded"
    return HealthResponse(status=status, model_loaded=model_loaded, version=settings.app_version)
