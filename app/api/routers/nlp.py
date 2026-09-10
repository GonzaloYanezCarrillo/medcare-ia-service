"""Routers de NLP (POST /nlp/sintomas, POST /nlp/resumen)."""

from fastapi import APIRouter, Depends

from app.schemas.nlp import (
    NlpResumenRequest,
    NlpResumenResponse,
    NlpSintomasRequest,
    NlpSintomasResponse,
)
from app.services.nlp_service import NlpService

router = APIRouter(prefix="/nlp", tags=["NLP"])


def get_nlp_service() -> NlpService:
    """Provider del servicio de NLP (inyección manual)."""
    return NlpService()


@router.post(
    "/sintomas",
    response_model=NlpSintomasResponse,
    summary="Extraer entidades de síntomas desde texto libre",
)
def extraer_sintomas(
    request: NlpSintomasRequest, service: NlpService = Depends(get_nlp_service)
) -> NlpSintomasResponse:
    """Procesa texto libre de síntomas (HU-NLP-01 / HU-NLP-02)."""
    return service.sintomas(request)


@router.post(
    "/resumen",
    response_model=NlpResumenResponse,
    summary="Generar resumen clínico y extraer entidades desde texto libre",
)
def generar_resumen(
    request: NlpResumenRequest, service: NlpService = Depends(get_nlp_service)
) -> NlpResumenResponse:
    """Genera el resumen clínico preliminar consumido por .NET al agendar."""
    return service.resumen(request)
