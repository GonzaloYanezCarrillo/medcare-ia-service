"""Schemas del feedback de asistencia real (contrato ia-api.yaml → POST /feedback/asistencia)."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Genero = Literal["M", "F", "Otro"]
CanalRecordatorio = Literal["email", "whatsapp", "ninguno"]


class FeedbackAsistenciaRequest(BaseModel):
    """Desenlace real + snapshot de features con que se predijo el riesgo (HU-IA-02).

    No incluye contenido clínico (RNF-SEG-03): solo features del PredictRequest,
    la fecha (para derivar Weekday) y la etiqueta `asistio`.
    """

    cita_id: str
    asistio: bool
    fecha_cita: date = Field(..., description="Fecha de la cita (YYYY-MM-DD); deriva Weekday")
    edad: int = Field(..., ge=0, le=120)
    genero: Genero
    dias_espera: int = Field(..., ge=0)
    especialidad: str
    ausencias_previas: int = Field(..., ge=0)
    canal_recordatorio: CanalRecordatorio = "ninguno"
    score_riesgo: float | None = Field(
        None, ge=0, le=1, description="Score predicho para esta cita (opcional)"
    )


class FeedbackAsistenciaResponse(BaseModel):
    """Respuesta de POST /feedback/asistencia."""

    cita_id: str
    asistio: bool
    registros_totales: int
