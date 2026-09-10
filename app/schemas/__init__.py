"""Schemas Pydantic del microservicio de IA.

Definiciones alineadas con `contracts/ia-api.yaml` (versión 1.1.1).
"""

from app.schemas.health import HealthResponse
from app.schemas.nlp import (
    EntidadSintoma,
    NlpResumenRequest,
    NlpResumenResponse,
    NlpSintomasRequest,
    NlpSintomasResponse,
)
from app.schemas.predict import PredictRequest, PredictResponse

__all__ = [
    "EntidadSintoma",
    "HealthResponse",
    "NlpResumenRequest",
    "NlpResumenResponse",
    "NlpSintomasRequest",
    "NlpSintomasResponse",
    "PredictRequest",
    "PredictResponse",
]
