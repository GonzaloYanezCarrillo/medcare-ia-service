"""Servicio NLP: extracción de entidades y resumen preliminar.

C2-2: motor spaCy (es_core_news_sm) para lematización/normalización sobre la
heurística de marcado de C0-1. La extracción se reemplaza por componentes
estadísticos de spaCy de forma incremental (C2-2 → C3-1).
"""

from __future__ import annotations

import re
from functools import lru_cache

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


@lru_cache(maxsize=1)
def _get_nlp():
    """Carga (una vez) el pipeline spaCy en español, con carga diferida.

    Solo se importa spaCy la primera vez que se invoca: las rutas que no usan NLP
    (/predict, /health) no pagan el costo de importación ni de modelo.
    """
    import spacy

    return spacy.load("es_core_news_sm")


def _lemmatizar(texto: str) -> str:
    """Normaliza texto a minúsculas con lemas (p. ej. 'días' → 'día', 'intensos' → 'intenso')."""
    if not texto.strip():
        return ""
    return " ".join(tok.lemma_ for tok in _get_nlp()(texto.lower()))


def _extraer_entidades(texto: str) -> list[EntidadSintoma]:
    entidades: list[EntidadSintoma] = []
    texto_l = texto.lower().strip()

    match = _DURACION_PATTERN.search(texto_l)
    if match:
        duracion = match.group(0).strip()
        entidades.append(EntidadSintoma(tipo="duracion", texto=duracion, severidad=None))

    # Síntomas: fragmentos separados por comas/semicolons/conectores (heurística básica).
    # El matching de severidad usa también el texto lematizado para reconocer
    # variaciones morfológicas ('graveS', 'intensOS', ...).
    sintomas = re.split(r",|;|\by\b|tambi[né]en", texto_l)
    for s in sintomas:
        s = s.strip(" .")
        if not s or re.fullmatch(r"desde hace.*|por .*|seguido.*", s):
            continue
        s_lemma = _lemmatizar(s) if s else ""
        if _contiene_keyword(s_lemma, _SEVERIDAD_KEYWORDS["alto"]) or _contiene_keyword(
            s, _SEVERIDAD_KEYWORDS["alto"]
        ):
            entidades.append(EntidadSintoma(tipo="sintoma", texto=s, severidad="alto"))
        elif _contiene_keyword(s_lemma, _SEVERIDAD_KEYWORDS["medio"]) or _contiene_keyword(
            s, _SEVERIDAD_KEYWORDS["medio"]
        ):
            entidades.append(EntidadSintoma(tipo="sintoma", texto=s, severidad="medio"))
        else:
            entidades.append(EntidadSintoma(tipo="sintoma", texto=s, severidad="bajo"))

    return entidades


def _contiene_keyword(texto: str, keywords: list[str]) -> bool:
    """Comprueba si algún keyword del listado aparece como palabra/texto dentro de `texto`."""
    return any(k in texto for k in keywords)


def _urgencia_sugerida(texto: str) -> str:
    texto_l = texto.lower()
    texto_lemma = _lemmatizar(texto) if texto_l else ""
    if _contiene_keyword(texto_lemma, _URGENCIA_KEYWORDS["alta"]) or _contiene_keyword(
        texto_l, _URGENCIA_KEYWORDS["alta"]
    ):
        return "alta"
    if _contiene_keyword(texto_lemma, _URGENCIA_KEYWORDS["media"]) or _contiene_keyword(
        texto_l, _URGENCIA_KEYWORDS["media"]
    ):
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
