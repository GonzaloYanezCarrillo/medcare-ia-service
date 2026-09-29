"""Tests de NLP (contrato ia-api.yaml → POST /nlp/sintomas, POST /nlp/resumen)."""

from app.services.nlp_service import (
    _extraer_alergias,
    _extraer_medicamentos,
    _lemmatizar,
    _urgencia_sugerida,
)
from tests.auth_utils import auth_headers

_CITA = "550e8400-e29b-41d4-a716-446655440000"


def test_nlp_sintomas_ok(client):
    response = client.post(
        "/nlp/sintomas",
        json={"texto_sintomas": "Dolor de cabeza desde hace 3 días, con visión borrosa"},
        headers=auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["entidades_extraidas"], list)
    assert body["urgencia_sugerida"] in {"baja", "media", "alta"}


def test_nlp_sintomas_sin_texto_422(client):
    response = client.post("/nlp/sintomas", json={}, headers=auth_headers())
    assert response.status_code == 422


def test_nlp_resumen_ok(client):
    response = client.post(
        "/nlp/resumen",
        json={
            "texto_sintomas": "Dolor de cabeza desde hace 3 días",
            "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        },
        headers=auth_headers(),
    )
    assert response.status_code == 200
    body = response.json()
    assert body["resumen"]
    assert isinstance(body["entidades_extraidas"], list)
    assert body["urgencia_sugerida"] in {"baja", "media", "alta"}


def test_lemmatizar_normaliza_variaciones():
    assert _lemmatizar("díAs intensOS graveS") == "día intenso grave"
    assert _lemmatizar("") == ""


def test_severidad_reconoce_variaciones_morfologicas(client):
    """C2-2: el matching de severidad usa lemas (antes 'intensos' no se detectaba)."""
    response = client.post(
        "/nlp/sintomas",
        json={"texto_sintomas": "Dolor de cabeza intensos desde hace 3 días"},
        headers=auth_headers(),
    )
    assert response.status_code == 200
    severidades = [e["severidad"] for e in response.json()["entidades_extraidas"]]
    assert "alto" in severidades


def test_urgencia_reconoce_forma_lematizada():
    """C2-2: 'convulsiones' (plural/sin tilde) no casaba con 'convulsión'; el lema sí."""
    assert _urgencia_sugerida("El paciente refiere convulsiones") == "alta"


def test_resumen_extrae_medicamento():
    """C3-1: los medicamentos declarados en el contrato se extraen sin ruido."""
    medicamentos = _extraer_medicamentos("Dolor de cabeza y tomo paracetamol hace 2 días")
    assert medicamentos == ["paracetamol"]
    assert "insulina" in _extraer_medicamentos("Toma insulina tres veces al día")


def test_resumen_extrae_alergia():
    """C3-1: las alergias se detectan por marcador y se limpia el artículo."""
    assert _extraer_alergias("soy alérgico a la penicilina") == ["penicilina"]
    assert _extraer_alergias("alergia a mariscos") == ["mariscos"]


def test_resumen_es_ficha_clinica_estructurada(client):
    """C3-1: el campo `resumen` es una ficha legible con contexto de la cita."""
    response = client.post(
        "/nlp/resumen",
        json={"texto_sintomas": "Fiebre alta desde hace 2 días", "cita_id": _CITA},
        headers=auth_headers(),
    )
    resumen = response.json()["resumen"]
    assert f"Ficha clínica preliminar — Cita {_CITA}" in resumen
    assert "Síntomas" in resumen
    assert "Duración:" in resumen
    assert "Urgencia sugerida:" in resumen
