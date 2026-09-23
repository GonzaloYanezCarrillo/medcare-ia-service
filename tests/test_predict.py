"""Tests de POST /predict (contrato ia-api.yaml)."""

from app.services.predict_service import _score_to_clase


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
    assert body["clase"] in {"asiste", "no_asiste"}


def test_predict_clase_coherente_con_score(client):
    """La clase discreta debe derivar del score con el umbral de decisión (0.5)."""
    response = client.post("/predict", json=_payload(ausencias_previas=8, dias_espera=40))
    body = response.json()
    assert body["score_riesgo"] >= 0.5
    assert body["clase"] == "no_asiste"


def test_predict_validacion_422(client):
    response = client.post("/predict", json={"cita_id": "x"})
    assert response.status_code == 422


def test_score_to_clase_umbral():
    assert _score_to_clase(0.49) == "asiste"
    assert _score_to_clase(0.5) == "no_asiste"
    assert _score_to_clase(0.9) == "no_asiste"
