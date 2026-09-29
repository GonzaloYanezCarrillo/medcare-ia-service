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

# Medicamentos conocidos (heurística C3-1): si aparecen en el texto, se marcan como
# medicamento aunque no vayan precedidos de un verbo ("toma paracetamol").
_MEDICAMENTOS_CONOCIDOS = [
    "paracetamol",
    "ibuprofeno",
    "amoxicilina",
    "aspirina",
    "omeprazol",
    "metformina",
    "losartán",
    "enalapril",
    "salbutamol",
    "amitriptilina",
    "insulina",
]

# Marcadores de medicación: capturan la palabra siguiente al verbo/conector.
_MEDICACION_PATTERN = re.compile(
    r"(?:toma|tomo|tomando|tome|medicad[oa]\s+con|medicamento|recetaron|est[aá]\s+con)\s+"
    r"([a-záéíóúüñ]+)",
    re.IGNORECASE,
)

# Marcadores de alergia: "alérgico a X", "alergia a X". Captura artículo+nombre
# (p. ej. "la penicilina") y lo limpia después en _extraer_alergias.
_ALERGIA_PATTERN = re.compile(
    r"(?:al[eé]rgic[oa]\s+a|alergias?\s+a)\s+"
    r"((?:(?:el|la|los|las|un|una|unos|unas)\s+)?[a-záéíóúüñ]+)",
    re.IGNORECASE,
)


def _normalizar_candidato(candidato: str) -> str:
    """Limpia un candidato extraído: quita signos y plurales fuera de la lista."""
    candidato = candidato.strip(".,; ")
    # El lema convierte plurales/sufijos a la forma base (p. ej. 'penicilina').
    try:
        return _lemmatizar(candidato) or candidato
    except Exception:  # noqa: BLE001
        return candidato


def _extraer_medicamentos(texto: str) -> list[str]:
    """Detecta medicamentos: por lista conocida o por marcador de medicación."""
    texto_l = texto.lower()
    hallazgos: list[str] = []
    for medicamento in _MEDICAMENTOS_CONOCIDOS:
        if medicamento in texto_l:
            hallazgos.append(medicamento)
    for match in _MEDICACION_PATTERN.finditer(texto_l):
        candidato = _normalizar_candidato(match.group(1))
        if candidato and candidato not in hallazgos:
            hallazgos.append(candidato)
    return hallazgos


_ARTICULOS = {"el", "la", "los", "las", "un", "una", "unos", "unas"}


def _extraer_alergias(texto: str) -> list[str]:
    """Detecta alergias por marcadores ('alérgico a X'), sin duplicar.

    Captura el sustantivo (con su artículo) y lo limpia sin depender de spaCy,
    para no acoplar la extracción al modelo de lematización.
    """
    hallazgos: list[str] = []
    for match in _ALERGIA_PATTERN.finditer(texto.lower()):
        partes = [p for p in match.group(1).split() if p not in _ARTICULOS]
        candidato = " ".join(partes)
        if candidato and candidato not in hallazgos:
            hallazgos.append(candidato)
    return hallazgos


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

    # Medicamentos y alergias (tipos ya declarados en el contrato, ahora extraídos).
    for medicamento in _extraer_medicamentos(texto_l):
        entidades.append(EntidadSintoma(tipo="medicamento", texto=medicamento, severidad=None))
    for alergia in _extraer_alergias(texto_l):
        entidades.append(EntidadSintoma(tipo="alergia", texto=alergia, severidad=None))

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
        """Genera resumen clínico preliminar (POST /nlp/resumen, HU-NLP-03)."""
        entidades = _extraer_entidades(request.texto_sintomas)
        urgencia = _urgencia_sugerida(request.texto_sintomas)
        return NlpResumenResponse(
            resumen=_build_ficha_clinica(request, entidades, urgencia),
            entidades_extraidas=entidades,
            urgencia_sugerida=urgencia,
        )


def _build_ficha_clinica(
    request: NlpResumenRequest, entidades: list[EntidadSintoma], urgencia: str
) -> str:
    """Construye la ficha clínica preliminar legible para el médico (C3-1).

    Estructura mínima en texto plano (el contrato expone solo `resumen: str`):
    identificación de la cita, síntomas con severidad, duración, medicamentos,
    alergias y urgencia sugerida.
    """
    by_tipo: dict[str, list[EntidadSintoma]] = {}
    for entidad in entidades:
        by_tipo.setdefault(entidad.tipo, []).append(entidad)

    lineas = [f"Ficha clínica preliminar — Cita {request.cita_id}", ""]

    if sintomas := by_tipo.get("sintoma"):
        lineas.append(f"Síntomas ({len(sintomas)}):")
        for sintoma in sintomas:
            sev = f" (severidad {sintoma.severidad})" if sintoma.severidad else ""
            lineas.append(f"  - {sintoma.texto}{sev}")

    if duraciones := by_tipo.get("duracion"):
        lineas.append("Duración: " + ", ".join(d.texto for d in duraciones))

    if medicamentos := by_tipo.get("medicamento"):
        lineas.append("Medicamentos: " + ", ".join(m.texto for m in medicamentos))

    if alergias := by_tipo.get("alergia"):
        lineas.append("Alergias: " + ", ".join(a.texto for a in alergias))

    if not sintomas and not duraciones and not medicamentos and not alergias:
        lineas.append("Sin hallazgos estructurados detectados a partir del relato.")

    lineas.append("")
    lineas.append(f"Urgencia sugerida: {urgencia}.")
    return "\n".join(lineas)
