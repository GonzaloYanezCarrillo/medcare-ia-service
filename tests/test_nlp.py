"""Tests de NLP (contrato ia-api.yaml → POST /nlp/sintomas, POST /nlp/resumen)."""

from app.services.nlp_service import _lemmatizar, _urgencia_sugerida


def test_nlp_sintomas_ok(client):
    response = client.post(
        "/nlp/sintomas",
        json={"texto_sintomas": "Dolor de cabeza desde hace 3 días, con visión borrosa"},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["entidades_extraidas"], list)
    assert body["urgencia_sugerida"] in {"baja", "media", "alta"}


def test_nlp_sintomas_sin_texto_422(client):
    response = client.post("/nlp/sintomas", json={})
    assert response.status_code == 422


def test_nlp_resumen_ok(client):
    response = client.post(
        "/nlp/resumen",
        json={
            "texto_sintomas": "Dolor de cabeza desde hace 3 días",
            "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        },
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
    )
    assert response.status_code == 200
    severidades = [e["severidad"] for e in response.json()["entidades_extraidas"]]
    assert "alto" in severidades


def test_urgencia_reconoce_forma_lematizada():
    """C2-2: 'convulsiones' (plural/sin tilde) no casaba con 'convulsión'; el lema sí."""
    assert _urgencia_sugerida("El paciente refiere convulsiones") == "alta"
