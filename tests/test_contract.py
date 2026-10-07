"""Pruebas de contrato de la API IA (C5-2).

Validan que la API cumple con `ia-api.yaml` usando TestClient:
- Esquema OpenAPI emitido válido y coherente con el contrato.
- Estructura y validaciones de requests/responses por endpoint.
- Códigos de error esperados (401/403/422) y seguridad (health sin auth).
"""

from __future__ import annotations

import time

import pytest
import yaml
from app.contract_path import resolver_contrato
from fastapi.testclient import TestClient
from tests.auth_utils import auth_headers

CONTRATO_PATH = resolver_contrato()


def _load_contrato() -> dict:
    with open(CONTRATO_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def contrato() -> dict:
    return _load_contrato()


def test_openapi_emitido_valido(client: TestClient, contrato: dict) -> None:
    """El /openapi.json emitido debe ser un OpenAPI válido y contener rutas del contrato."""
    r = client.get("/openapi.json")
    assert r.status_code == 200
    spec = r.json()

    # openapi-spec-validator opcional
    try:
        from openapi_spec_validator import validate_spec  # type: ignore

        validate_spec(spec)
    except Exception as exc:  # pragma: no cover - solo validación
        pytest.fail(f"Spec OpenAPI inválido: {exc}")

    paths_contrato = set((contrato.get("paths") or {}).keys())
    paths_spec = set((spec.get("paths") or {}).keys())
    assert paths_contrato.issubset(paths_spec), f"Faltan rutas: {paths_contrato - paths_spec}"


def test_health_sin_auth(client: TestClient) -> None:
    """GET /health debe ser público (security: [])."""
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"healthy", "degraded", "unhealthy"}
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["version"], str) and body["version"]


def test_nlp_sintomas_extraccion_campos(client: TestClient) -> None:
    """POST /nlp/sintomas: estructura de respuesta según contrato."""
    payload = {"texto_sintomas": "Dolor de cabeza desde hace 3 días, con visión borrosa"}
    r = client.post("/nlp/sintomas", json=payload, headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert "entidades_extraidas" in body
    assert isinstance(body["entidades_extraidas"], list)
    for e in body["entidades_extraidas"]:
        assert e["tipo"] in {"sintoma", "duracion", "severidad", "medicamento", "alergia"}
        assert isinstance(e["texto"], str)
        if "severidad" in e and e["severidad"] is not None:
            assert e["severidad"] in {"bajo", "medio", "alto"}
    if body.get("urgencia_sugerida") is not None:
        assert body["urgencia_sugerida"] in {"baja", "media", "alta"}


def test_nlp_sintomas_requiere_texto(client: TestClient) -> None:
    """Falta texto_sintomas → 422."""
    r = client.post("/nlp/sintomas", json={}, headers=auth_headers())
    assert r.status_code == 422


def test_nlp_sintomas_con_cita_id_opcional(client: TestClient) -> None:
    """cita_id opcional no debe romper validación."""
    payload = {
        "texto_sintomas": "Tengo dolor abdominal leve",
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
    }
    r = client.post("/nlp/sintomas", json=payload, headers=auth_headers())
    assert r.status_code == 200


def test_nlp_resumen_campos_requeridos(client: TestClient) -> None:
    """POST /nlp/resumen exige texto_sintomas y cita_id."""
    # Falta cita_id
    r = client.post("/nlp/resumen", json={"texto_sintomas": "dolor"}, headers=auth_headers())
    assert r.status_code == 422
    # Falta texto_sintomas
    r = client.post(
        "/nlp/resumen",
        json={"cita_id": "550e8400-e29b-41d4-a716-446655440000"},
        headers=auth_headers(),
    )
    assert r.status_code == 422


def test_nlp_resumen_respuesta(client: TestClient) -> None:
    """Estructura de /nlp/resumen según contrato."""
    payload = {
        "texto_sintomas": "Tengo dolor de cabeza fuerte desde hace 3 días y visión borrosa",
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
    }
    r = client.post("/nlp/resumen", json=payload, headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body.get("resumen"), str) and body["resumen"]
    assert isinstance(body.get("entidades_extraidas"), list)
    if body.get("urgencia_sugerida") is not None:
        assert body["urgencia_sugerida"] in {"baja", "media", "alta"}


def test_predict_campos_y_coherencias(client: TestClient) -> None:
    """POST /predict: campos requeridos, enums, rangos y coherencias."""
    payload = {
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        "edad": 34,
        "genero": "F",
        "dias_espera": 5,
        "especialidad": "psicologia",
        "ausencias_previas": 1,
        "canal_recordatorio": "whatsapp",
    }
    r = client.post("/predict", json=payload, headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["cita_id"] == payload["cita_id"]
    score = body["score_riesgo"]
    assert isinstance(score, (int, float))
    assert 0 <= score <= 1
    assert body["banda_riesgo"] in {"Bajo", "Medio", "Alto"}
    assert body["clase"] in {"asiste", "no_asiste"}
    # Coherencias (mismas reglas que contrato/implementación)
    banda = body["banda_riesgo"]
    banda_ok = (
        (score < 0.25 and banda == "Bajo")
        or (0.25 <= score <= 0.50 and banda == "Medio")
        or (score > 0.50 and banda == "Alto")
    )
    assert banda_ok
    clase_ok = (score >= 0.5 and body["clase"] == "no_asiste") or (
        score < 0.5 and body["clase"] == "asiste"
    )
    assert clase_ok


@pytest.mark.parametrize(
    "faltante",
    [
        "cita_id",
        "edad",
        "genero",
        "dias_espera",
        "especialidad",
        "ausencias_previas",
    ],
)
def test_predict_faltan_campos_requeridos(client: TestClient, faltante: str) -> None:
    """Cada campo requerido debe provocar 422."""
    payload = {
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        "edad": 30,
        "genero": "M",
        "dias_espera": 2,
        "especialidad": "medicina_general",
        "ausencias_previas": 0,
    }
    payload.pop(faltante)
    r = client.post("/predict", json=payload, headers=auth_headers())
    assert r.status_code == 422


def test_predict_enums(client: TestClient) -> None:
    """Valores fuera de enum deben dar 422."""
    # genero inválido
    payload = {
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        "edad": 40,
        "genero": "X",
        "dias_espera": 1,
        "especialidad": "dental",
        "ausencias_previas": 0,
    }
    r = client.post("/predict", json=payload, headers=auth_headers())
    assert r.status_code == 422
    # canal_recordatorio inválido
    payload["genero"] = "F"
    payload["canal_recordatorio"] = "sms"
    r = client.post("/predict", json=payload, headers=auth_headers())
    assert r.status_code == 422


def test_feedback_asistencia_estructura_contrato(client: TestClient) -> None:
    """POST /feedback/asistencia: campos requeridos de FeedbackAsistenciaResponse."""
    payload = {
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
    r = client.post("/feedback/asistencia", json=payload, headers=auth_headers())
    assert r.status_code == 200
    body = r.json()
    assert body["cita_id"] == payload["cita_id"]
    assert isinstance(body["asistio"], bool)
    assert isinstance(body["registros_totales"], int) and body["registros_totales"] >= 1


def test_feedback_asistencia_requiere_body(client: TestClient) -> None:
    """Sin body o con campos requeridos ausentes → 422 (validación Pydantic)."""
    r = client.post("/feedback/asistencia", json={}, headers=auth_headers())
    assert r.status_code == 422
    r = client.post("/feedback/asistencia", headers=auth_headers())
    assert r.status_code == 422


def test_auth_sin_token_401(client: TestClient) -> None:
    """Rutas protegidas sin token → 401."""
    r = client.post("/predict", json={"edad": 1}, headers={})
    assert r.status_code == 401
    r = client.post("/nlp/sintomas", json={"texto_sintomas": "x"}, headers={})
    assert r.status_code == 401
    r = client.post(
        "/nlp/resumen",
        json={
            "texto_sintomas": "x",
            "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        },
        headers={},
    )
    assert r.status_code == 401
    r = client.post("/feedback/asistencia", json={"asistio": True}, headers={})
    assert r.status_code == 401


def test_auth_rol_invalido_403(client: TestClient) -> None:
    """Rol distinto a service-ia → 403 (fail-closed por política)."""
    r = client.post(
        "/predict",
        json={
            "cita_id": "550e8400-e29b-41d4-a716-446655440000",
            "edad": 20,
            "genero": "M",
            "dias_espera": 1,
            "especialidad": "dental",
            "ausencias_previas": 0,
        },
        headers=auth_headers(role="otro-rol"),
    )
    assert r.status_code == 403
    r = client.post(
        "/feedback/asistencia",
        json={
            "cita_id": "550e8400-e29b-41d4-a716-446655440000",
            "asistio": True,
            "fecha_cita": "2026-10-05",
            "edad": 30,
            "genero": "F",
            "dias_espera": 3,
            "especialidad": "dental",
            "ausencias_previas": 0,
        },
        headers=auth_headers(role="otro-rol"),
    )
    assert r.status_code == 403


def test_latencia_predict(client: TestClient) -> None:
    """Latencia de inferencia /predict (umbral orientativo para contrato/RNF)."""
    payload = {
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        "edad": 34,
        "genero": "F",
        "dias_espera": 5,
        "especialidad": "psicologia",
        "ausencias_previas": 1,
    }
    t0 = time.perf_counter()
    r = client.post("/predict", json=payload, headers=auth_headers())
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200
    # Umbral conservador (modelo ligero en tests): < 800ms
    assert dt_ms < 800.0, f"/predict {dt_ms:.1f}ms >= 800ms"


def test_latencia_nlp_resumen(client: TestClient) -> None:
    """Latencia de NLP /nlp/resumen (spaCy ligero)."""
    payload = {
        "texto_sintomas": "Tengo dolor de cabeza fuerte desde hace 3 días y visión borrosa",
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
    }
    t0 = time.perf_counter()
    r = client.post("/nlp/resumen", json=payload, headers=auth_headers())
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert r.status_code == 200
    assert dt_ms < 1500.0, f"/nlp/resumen {dt_ms:.1f}ms >= 1500ms"
