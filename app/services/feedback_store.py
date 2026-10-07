"""Persistencia del feedback de asistencia real (C4-1 / HU-IA-02).

Almacena el desenlace de cada cita junto al snapshot de features con que se predijo,
en un JSONL con upsert idempotente por `cita_id`: reenviar la misma cita sobrescribe
el registro anterior (el último estado gana).

RNF-SEG-03: solo se persisten features + etiqueta del reentrenamiento; nunca
contenido clínico ni identificadores personales más allá de `cita_id`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings

logger = logging.getLogger(__name__)

# Whitelist de campos persistidos (RNF-SEG-03): todo lo demás se descarta.
CAMPOS_PERMITIDOS: frozenset[str] = frozenset(
    {
        "cita_id",
        "asistio",
        "fecha_cita",
        "edad",
        "genero",
        "dias_espera",
        "especialidad",
        "ausencias_previas",
        "canal_recordatorio",
        "score_riesgo",
    }
)


class FeedbackStore:
    """JSONL append-upsert por `cita_id` (último estado gana)."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def registrar(self, registro: dict[str, Any]) -> tuple[dict[str, Any], int]:
        """Registra (o actualiza) el desenlace de una cita.

        Returns:
            (registro persistido, total de registros acumulados)
        """
        limpio = {k: v for k, v in registro.items() if k in CAMPOS_PERMITIDOS}
        limpio["registrado_en"] = datetime.now(UTC).isoformat(timespec="seconds")
        with self._lock:
            registros = self._cargar()
            registros[str(limpio["cita_id"])] = limpio
            self._guardar(registros)
        logger.info(
            "Feedback registrado para cita %s (total=%d)", limpio["cita_id"], len(registros)
        )
        return limpio, len(registros)

    def registros(self) -> list[dict[str, Any]]:
        """Todos los registros (para reentrenamiento/export)."""
        with self._lock:
            return list(self._cargar().values())

    def total(self) -> int:
        with self._lock:
            return len(self._cargar())

    def _cargar(self) -> dict[str, dict[str, Any]]:
        if not self._path.exists():
            return {}
        registros: dict[str, dict[str, Any]] = {}
        with open(self._path, encoding="utf-8") as f:
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    registro = json.loads(linea)
                except json.JSONDecodeError:
                    logger.warning("Linea corrupta ignorada en %s", self._path)
                    continue
                if "cita_id" in registro:
                    registros[str(registro["cita_id"])] = registro
        return registros

    def _guardar(self, registros: dict[str, dict[str, Any]]) -> None:
        """Escritura atómica (tmp + rename) para no corromper el JSONL."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for registro in registros.values():
                f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        os.replace(tmp, self._path)


@lru_cache
def get_feedback_store() -> FeedbackStore:
    """Singleton del store de feedback (path desde `Settings.feedback_path`)."""
    settings = get_settings()
    return FeedbackStore(settings.feedback_path)
