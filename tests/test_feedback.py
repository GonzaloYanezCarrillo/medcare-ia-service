"""Tests del endpoint POST /feedback/asistencia (C4-1 / HU-IA-02)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.auth_utils import auth_headers

PAYLOAD = {
    "cita_id": "550e8400-e29b-41d4-a716-446655440000",
    "asistio": False,
    "fecha_cita": "2026-10-05",
    "edad": 34,
    "genero": "F",
    "dias_espera": 5,
    "especialidad": "psicologia",
    "ausencias_previas": 1,
    "canal_recordatorio": "whatsapp",
    "score_riesgo": 0.63,
}


def _post(client: TestClient, payload: dict, **kw) -> object:
    return client.post("/feedback/asistencia", json=payload, headers=auth_headers(**kw))


def test_feedback_registra_y_responde(client: TestClient, feedback_store) -> None:
    r = _post(client, PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["cita_id"] == PAYLOAD["cita_id"]
    assert body["asistio"] is False
    assert body["registros_totales"] == 1
    (registro,) = feedback_store.registros()
    assert registro["asistio"] is False
    assert registro["fecha_cita"] == "2026-10-05"
    assert registro["score_riesgo"] == 0.63


def test_feedback_asistio_true(client: TestClient, feedback_store) -> None:
    r = _post(client, {**PAYLOAD, "asistio": True})
    assert r.status_code == 200
    assert r.json()["asistio"] is True


def test_feedback_upsert_idempotente(client: TestClient, feedback_store) -> None:
    _post(client, PAYLOAD)
    r = _post(client, {**PAYLOAD, "asistio": True})
    assert r.status_code == 200
    assert r.json()["registros_totales"] == 1
    assert feedback_store.total() == 1
    (registro,) = feedback_store.registros()
    assert registro["asistio"] is True


def test_feedback_campo_score_opcional(client: TestClient) -> None:
    payload = {k: v for k, v in PAYLOAD.items() if k != "score_riesgo"}
    r = _post(client, payload)
    assert r.status_code == 200


def test_feedback_canal_default(client: TestClient, feedback_store) -> None:
    payload = {k: v for k, v in PAYLOAD.items() if k != "canal_recordatorio"}
    r = _post(client, payload)
    assert r.status_code == 200
    (registro,) = feedback_store.registros()
    assert registro["canal_recordatorio"] == "ninguno"


def test_feedback_sin_token_401(client: TestClient) -> None:
    r = client.post("/feedback/asistencia", json=PAYLOAD, headers={})
    assert r.status_code == 401


def test_feedback_rol_invalido_403(client: TestClient) -> None:
    r = _post(client, PAYLOAD, role="recepcionista")
    assert r.status_code == 403


@pytest.mark.parametrize(
    "faltante",
    [
        "cita_id",
        "asistio",
        "fecha_cita",
        "edad",
        "genero",
        "dias_espera",
        "especialidad",
        "ausencias_previas",
    ],
)
def test_feedback_campos_requeridos_422(client: TestClient, faltante: str) -> None:
    payload = {k: v for k, v in PAYLOAD.items() if k != faltante}
    r = _post(client, payload)
    assert r.status_code == 422


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("genero", "X"),
        ("canal_recordatorio", "sms"),
        ("edad", -1),
        ("dias_espera", -5),
        ("ausencias_previas", -1),
        ("score_riesgo", 1.5),
        ("fecha_cita", "no-es-fecha"),
        ("asistio", "si"),
    ],
)
def test_feedback_valores_invalidos_422(client: TestClient, campo, valor) -> None:
    r = _post(client, {**PAYLOAD, campo: valor})
    assert r.status_code == 422


def test_feedback_descarta_contenido_clinico(client: TestClient, feedback_store) -> None:
    """RNF-SEG-03: campos clínicos extra no llegan al store."""
    r = _post(
        client,
        {**PAYLOAD, "sintomas": "dolor torácico", "nombre_paciente": "Juan Pérez"},
    )
    assert r.status_code == 200
    (registro,) = feedback_store.registros()
    assert "sintomas" not in registro
    assert "nombre_paciente" not in registro
