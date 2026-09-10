"""Schemas NLP (contrato ia-api.yaml → POST /nlp/sintomas, POST /nlp/resumen)."""

from typing import Literal

from pydantic import BaseModel, Field

TipoEntidad = Literal["sintoma", "duracion", "severidad", "medicamento", "alergia"]
Severidad = Literal["bajo", "medio", "alto"]
Urgencia = Literal["baja", "media", "alta"]


class EntidadSintoma(BaseModel):
    """Entidad extraída del texto libre."""

    tipo: TipoEntidad
    texto: str
    severidad: Severidad | None = None


class NlpSintomasRequest(BaseModel):
    """Cuerpo de petición de POST /nlp/sintomas (HU-NLP-01 / HU-NLP-02)."""

    texto_sintomas: str = Field(..., min_length=1, description="Descripción libre de síntomas")
    cita_id: str | None = Field(default=None, description="Identificador opcional de la cita")


class NlpSintomasResponse(BaseModel):
    """Respuesta de POST /nlp/sintomas: entidades extraídas."""

    entidades_extraidas: list[EntidadSintoma]
    urgencia_sugerida: Urgencia | None = None


class NlpResumenRequest(BaseModel):
    """Cuerpo de petición de POST /nlp/resumen."""

    texto_sintomas: str = Field(..., min_length=1)
    cita_id: str


class NlpResumenResponse(BaseModel):
    """Respuesta de POST /nlp/resumen: ficha clínica preliminar."""

    resumen: str
    entidades_extraidas: list[EntidadSintoma]
    urgencia_sugerida: Urgencia
