"""Servicio NLP: extracción de entidades y resumen preliminar.

Implementación de marcado (heurística) para validar el contrato en C0-1.
Se reemplaza/aumenta con spaCy/NLTK en los sprints de NLP (C2-2, C3-1).
"""

import re

from app.schemas.nlp import (
    EntidadSintoma,
    NlpResumenRequest,
    NlpResumenResponse,
    NlpSintomasRequest,
    NlpSintomasResponse,
)

_DURACION_PATTERN = re.compile(
    r"(?P<dias>\d+)\s*d[ií]as?|"
    r"(?P<semanas>\d+)\s*semanas?|"
    r"(?P<meses>\d+)\s*mes(es)?|"
    r"(?P<horas>\d+)\s*horas?",
    re.IGNORECASE,
)

_SEVERIDAD_KEYWORDS = {
    "alto": ["intenso", "grave", "severo", "insoportable", "fuerte", "agudo"],
    "medio": ["moderado", "considerable"],
}

_URGENCIA_KEYWORDS = {
    "alta": ["visión borrosa", "desmayo", "convulsión", "falta de aire", "sangrado", "pecho"],
    "media": ["fiebre", "nauseas", "vómitos", "mareo"],
}


def _extraer_entidades(texto: str) -> list[EntidadSintoma]:
    entidades: list[EntidadSintoma] = []
    texto_l = texto.lower().strip()

    match = _DURACION_PATTERN.search(texto_l)
    if match:
        duracion = match.group(0).strip()
        entidades.append(EntidadSintoma(tipo="duracion", texto=duracion, severidad=None))

    # Síntomas: fragmentos separados por comas/semicolons/conectores (heurística básica).
    sintomas = re.split(r",|;|\by\b|tambi[né]en", texto_l)
    for s in sintomas:
        s = s.strip(" .")
        if not s or re.fullmatch(r"desde hace.*|por .*|seguido.*", s):
            continue
        if any(k in s for k in _SEVERIDAD_KEYWORDS["alto"]):
            entidades.append(EntidadSintoma(tipo="sintoma", texto=s, severidad="alto"))
        elif any(k in s for k in _SEVERIDAD_KEYWORDS["medio"]):
            entidades.append(EntidadSintoma(tipo="sintoma", texto=s, severidad="medio"))
        else:
            entidades.append(EntidadSintoma(tipo="sintoma", texto=s, severidad="bajo"))

    return entidades


def _urgencia_sugerida(texto: str) -> str:
    texto_l = texto.lower()
    if any(k in texto_l for k in _URGENCIA_KEYWORDS["alta"]):
        return "alta"
    if any(k in texto_l for k in _URGENCIA_KEYWORDS["media"]):
        return "media"
    return "baja"


class NlpService:
    """Procesamiento de lenguaje natural del servicio IA."""

    def sintomas(self, request: NlpSintomasRequest) -> NlpSintomasResponse:
        """Extrae entidades desde texto libre (POST /nlp/sintomas)."""
        entidades = _extraer_entidades(request.texto_sintomas)
        return NlpSintomasResponse(
            entidades_extraidas=entidades,
            urgencia_sugerida=_urgencia_sugerida(request.texto_sintomas),
        )

    def resumen(self, request: NlpResumenRequest) -> NlpResumenResponse:
        """Genera resumen clínico preliminar (POST /nlp/resumen)."""
        entidades = _extraer_entidades(request.texto_sintomas)
        urgencia = _urgencia_sugerida(request.texto_sintomas)
        resumen = (
            f"Consulta {request.cita_id}: {len(entidades)} hallazgo(s) identificado(s) "
            f"a partir del relato del paciente. Urgencia preliminar: {urgencia}."
        )
        return NlpResumenResponse(
            resumen=resumen,
            entidades_extraidas=entidades,
            urgencia_sugerida=urgencia,
        )
