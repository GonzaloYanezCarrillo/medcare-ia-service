"""Tests del FeedbackStore (C4-1 / HU-IA-02): JSONL upsert + RNF-SEG-03."""

from __future__ import annotations

import json
from pathlib import Path

from app.services.feedback_store import CAMPOS_PERMITIDOS, FeedbackStore

REGISTRO = {
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


def _store(tmp_path: Path) -> FeedbackStore:
    return FeedbackStore(tmp_path / "feedback" / "asistencia.jsonl")


def test_registrar_crea_archivo_y_total(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert store.total() == 0
    _, total = store.registrar(REGISTRO)
    assert total == 1
    assert store.path.exists()
    assert store.total() == 1


def test_upsert_idempotente_por_cita(tmp_path: Path) -> None:
    """Reenviar la misma cita no duplica: el último estado gana."""
    store = _store(tmp_path)
    store.registrar(REGISTRO)
    store.registrar({**REGISTRO, "asistio": True, "score_riesgo": 0.42})
    assert store.total() == 1
    (registro,) = store.registros()
    assert registro["asistio"] is True
    assert registro["score_riesgo"] == 0.42


def test_distintas_citas_acumulan(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.registrar(REGISTRO)
    store.registrar({**REGISTRO, "cita_id": "0a7b1c2d-0000-4000-8000-000000000001"})
    assert store.total() == 2


def test_whitelist_sin_contenido_clinico(tmp_path: Path) -> None:
    """RNF-SEG-03: campos clínicos/personales extra no se persisten."""
    store = _store(tmp_path)
    sucio = {
        **REGISTRO,
        "sintomas": "dolor torácico con disnea",
        "nombre_paciente": "Juan Pérez",
        "diagnostico": "IAM",
    }
    store.registrar(sucio)
    (registro,) = store.registros()
    assert "sintomas" not in registro
    assert "nombre_paciente" not in registro
    assert "diagnostico" not in registro
    assert set(registro) - {"registrado_en"} <= CAMPOS_PERMITIDOS


def test_whitelist_es_cerrada() -> None:
    assert "sintomas" not in CAMPOS_PERMITIDOS
    assert "texto_sintomas" not in CAMPOS_PERMITIDOS
    assert "nombre_paciente" not in CAMPOS_PERMITIDOS


def test_linea_corrupta_se_ignora(tmp_path: Path) -> None:
    store = _store(tmp_path)
    store.registrar(REGISTRO)
    with open(store.path, "a", encoding="utf-8") as f:
        f.write("{esto no es json\n")
    assert store.total() == 1
    store.registrar({**REGISTRO, "cita_id": "0a7b1c2d-0000-4000-8000-000000000002"})
    assert store.total() == 2


def test_persistencia_entre_instancias(tmp_path: Path) -> None:
    """El JSONL sobrevive a un nuevo store (simula reinicio/retrain)."""
    path = tmp_path / "asistencia.jsonl"
    FeedbackStore(path).registrar(REGISTRO)
    recargado = FeedbackStore(path)
    assert recargado.total() == 1
    (registro,) = recargado.registros()
    assert json.dumps(registro)  # serializable


def test_registrar_sella_timestamp(tmp_path: Path) -> None:
    store = _store(tmp_path)
    registro, _ = store.registrar(REGISTRO)
    assert "registrado_en" in registro
    assert registro["registrado_en"]  # ISO 8601 UTC
