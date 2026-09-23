"""Schemas de predicción de no-show (contrato ia-api.yaml → POST /predict)."""

from typing import Literal

from pydantic import BaseModel, Field

Genero = Literal["M", "F", "Otro"]
BandaRiesgo = Literal["Bajo", "Medio", "Alto"]
ClasePrediccion = Literal["asiste", "no_asiste"]
CanalRecordatorio = Literal["email", "whatsapp", "ninguno"]


class PredictRequest(BaseModel):
    """Entradas para calcular el riesgo de no-show de una cita."""

    cita_id: str
    edad: int = Field(..., ge=0, le=120)
    genero: Genero
    dias_espera: int = Field(
        ..., ge=0, description="Días entre el agendamiento y la cita (WaitingDays)"
    )
    especialidad: str
    ausencias_previas: int = Field(..., ge=0)
    canal_recordatorio: CanalRecordatorio = "ninguno"


class PredictResponse(BaseModel):
    """Respuesta de POST /predict."""

    cita_id: str
    score_riesgo: float = Field(..., ge=0, le=1)
    banda_riesgo: BandaRiesgo
    clase: ClasePrediccion
