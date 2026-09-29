"""Consumidor simulado del backend .NET contra el microservicio IA real.

Harness de integración I6-2: reproduce la llamada que el backend .NET (Dev B) hará
al microservicio de IA según el contrato `ia-api.yaml`:

- GET  /health          (sin auth)             -> si el servicio está sano
- POST /predict         (JWT M2M service-ia)   -> score + banda + clase de no-show
- POST /nlp/sintomas    (JWT M2M service-ia)   -> entidades de síntomas
- POST /nlp/resumen     (JWT M2M service-ia)   -> ficha clínica preliminar

Valida las respuestas contra el contrato (campos, tipos, enums y coherencias) e
informa como lo haría una integración real. Ejecutar con el servicio ya levantado:
    python scripts/integracion/consumidor_dotnet.py [base_url]
"""

from __future__ import annotations

import os
import sys

import httpx

sys.path.insert(0, os.path.dirname(__file__))
from m2m_keys import auth_headers  # noqa: E402

BASE_URL = (
    sys.argv[1] if len(sys.argv) > 1 else os.environ.get("IA_BASE_URL", "http://127.0.0.1:8100")
)

RESULTADO_ESPACIOS = "\n    de resultados: "


def _checket(nombre: str, condicion: bool, detalle: str = "") -> None:
    """Imprime el resultado de un paso de validación."""
    marca = "OK " if condicion else "FALLO"
    print(f"  [{marca}] {nombre}" + (f" — {detalle}" if detalle else ""))
    if not condicion:
        raise SystemExit(f"Paso fallido: {nombre}")


def paso_health(client: httpx.Client) -> None:
    print("PASO 1 — GET /health (sin token: contrato security: [])")
    r = client.get("/health")
    _checket("status 200", r.status_code == 200, f"recibido {r.status_code}")
    body = r.json()
    _checket(
        "status en [healthy, degraded, unhealthy]",
        body.get("status") in {"healthy", "degraded", "unhealthy"},
    )
    _checket("model_loaded es bool", isinstance(body.get("model_loaded"), bool))
    _checket("version es str", isinstance(body.get("version"), str) and body.get("version"))
    print(f"    -> {body}")
    print()


def paso_predict(client: httpx.Client) -> None:
    print("PASO 2 — POST /predict (payload real de una cita, JWT service-ia)")
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
    _checket("status 200", r.status_code == 200, f"recibido {r.status_code}")
    body = r.json()
    _checket("cita_id espejado", body.get("cita_id") == payload["cita_id"])
    score = body.get("score_riesgo")
    _checket(
        "score_riesgo en [0,1]",
        isinstance(score, (int, float)) and 0 <= score <= 1,
        f"score={score}",
    )
    _checket("banda_riesgo enum", body.get("banda_riesgo") in {"Bajo", "Medio", "Alto"})
    _checket("clase enum", body.get("clase") in {"asiste", "no_asiste"})

    # Coherencias del contrato: banda por umbrales y clase por umbral 0.5.
    banda = body["banda_riesgo"]
    banda_ok = (
        (score < 0.25 and banda == "Bajo")
        or (0.25 <= score <= 0.50 and banda == "Medio")
        or (score > 0.50 and banda == "Alto")
    )
    _checket("banda coherente con score", banda_ok, f"{banda} para score={score}")
    clase_ok = (score >= 0.5 and body["clase"] == "no_asiste") or (
        score < 0.5 and body["clase"] == "asiste"
    )
    _checket("clase coherente con score", clase_ok, f"{body['clase']} para score={score}")

    print(f"    -> {body}")
    print()


def paso_nlp_sintomas(client: httpx.Client) -> None:
    print("PASO 3 — POST /nlp/sintomas (texto libre, JWT service-ia)")
    r = client.post(
        "/nlp/sintomas",
        json={"texto_sintomas": "Dolor de cabeza desde hace 3 días, con visión borrosa"},
        headers=auth_headers(),
    )
    _checket("status 200", r.status_code == 200, f"recibido {r.status_code}")
    body = r.json()
    entidades = body.get("entidades_extraidas", [])
    _checket(
        "entidades_extraidas lista no vacía", isinstance(entidades, list) and len(entidades) > 0
    )
    tipos_validos = {"sintoma", "duracion", "severidad", "medicamento", "alergia"}
    _checket(
        "tipos de entidad dentro del enum del contrato",
        all(e.get("tipo") in tipos_validos for e in entidades),
    )
    severidades_validas = {"bajo", "medio", "alto"}
    _checket(
        "severidad (si viene) dentro del enum",
        all(
            e.get("severidad") in severidades_validas or e.get("severidad") is None
            for e in entidades
        ),
    )
    _checket("urgencia_sugerida enum", body.get("urgencia_sugerida") in {"baja", "media", "alta"})
    resumen = [(e["tipo"], e["texto"]) for e in entidades]
    print(f"    -> urgencia={body['urgencia_sugerida']}, entidades={resumen}")
    print()


def paso_nlp_resumen(client: httpx.Client) -> None:
    print("PASO 4 — POST /nlp/resumen (ficha clínica, JWT service-ia)")
    payload = {
        "texto_sintomas": "Tengo dolor de cabeza fuerte desde hace 3 días y visión borrosa",
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
    }
    r = client.post("/nlp/resumen", json=payload, headers=auth_headers())
    _checket("status 200", r.status_code == 200, f"recibido {r.status_code}")
    body = r.json()
    _checket("resumen no vacío y string", isinstance(body.get("resumen"), str) and body["resumen"])
    _checket("ficha incluye la cita", payload["cita_id"] in body["resumen"])
    _checket("ficha incluye sección Síntomas", "Síntomas" in body["resumen"])
    _checket(
        "ficha incluye Duración/Medicamentos/Alergias cuando aplica",
        any(s in body["resumen"] for s in ("Duración:", "Medicamentos:", "Alergias:")),
    )
    _checket("urgencia_sugerida enum", body.get("urgencia_sugerida") in {"baja", "media", "alta"})
    print("    -> " + body["resumen"].replace("\n", "\n    "))
    print()


def paso_errores(client: httpx.Client) -> None:
    print("PASO 5 — Manejo de errores HTTP (como lo ve .NET)")
    r = client.post(
        "/predict",
        json={
            "cita_id": "550e8400-e29b-41d4-a716-446655440000",
            "edad": 34,
            "genero": "F",
            "dias_espera": 5,
            "especialidad": "psicologia",
            "ausencias_previas": 1,
        },
    )
    _checket("sin token -> 401", r.status_code == 401, f"recibido {r.status_code}")

    r = client.post(
        "/predict",
        json={
            "cita_id": "550e8400-e29b-41d4-a716-446655440000",
            "edad": 34,
            "genero": "F",
            "dias_espera": 5,
            "especialidad": "psicologia",
            "ausencias_previas": 1,
        },
        headers=auth_headers(role="service-omro"),
    )
    _checket("rol no autorizado -> 403", r.status_code == 403, f"recibido {r.status_code}")

    r = client.post("/predict", json={"cita_id": "x"}, headers=auth_headers())
    _checket("payload inválido -> 422", r.status_code == 422, f"recibido {r.status_code}")
    print()


def main() -> None:
    print(f"Consumidor simulado .NET hacia {BASE_URL}\n")
    with httpx.Client(base_url=BASE_URL, timeout=30.0) as client:
        paso_health(client)
        paso_predict(client)
        paso_nlp_sintomas(client)
        paso_nlp_resumen(client)
        paso_errores(client)
    print("\nIntegración IA ↔ .NET OK: el microservicio responde según ia-api.yaml.")


if __name__ == "__main__":
    main()
