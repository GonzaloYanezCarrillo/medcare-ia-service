"""Tests de POST /predict (contrato ia-api.yaml)."""


def _payload(cita_id="550e8400-e29b-41d4-a716-446655440000", **overrides):
    base = {
        "cita_id": cita_id,
        "edad": 34,
        "genero": "F",
        "dias_espera": 5,
        "especialidad": "psicologia",
        "ausencias_previas": 1,
        "canal_recordatorio": "whatsapp",
    }
    base.update(overrides)
    return base


def test_predict_ok(client):
    response = client.post("/predict", json=_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["cita_id"] == _payload()["cita_id"]
    assert 0 <= body["score_riesgo"] <= 1
    assert body["banda_riesgo"] in {"Bajo", "Medio", "Alto"}


def test_predict_validacion_422(client):
    response = client.post("/predict", json={"cita_id": "x"})
    assert response.status_code == 422
