"""Tests de NLP (contrato ia-api.yaml → POST /nlp/sintomas, POST /nlp/resumen)."""


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
