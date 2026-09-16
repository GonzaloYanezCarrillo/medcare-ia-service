"""Registry del modelo entrenado.

Carga el artefacto `.joblib` desde `MODEL_PATH` (si existe) en el arranque, de modo que
`GET /health` y `POST /predict` reflejen su disponibilidad. Valida que la versión del
artefacto cumpla `MODEL_MIN_VERSION`; si no la cumple o el archivo no existe, el
servicio opera en modo "degraded" con un predictor de marcado.
"""

import logging
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Contenedor del artefacto entrenado (C1-1), con validación de versión."""

    def __init__(self, model_path: str, model_min_version: str = "0.1.0") -> None:
        self._path = Path(model_path)
        self._min_version = model_min_version
        self._model = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("Modelo no encontrado en %s. Servicio en modo degradado.", self._path)
            return
        try:
            import joblib

            artifact = joblib.load(self._path)
            if not isinstance(artifact, dict) or "pipeline" not in artifact:
                logger.error("Artefacto inválido en %s: falta 'pipeline'.", self._path)
                return
            version = str(artifact.get("model_version", "0.0.0"))
            if self._lower(version) < self._lower(self._min_version):
                logger.warning(
                    "Modelo %s (>=%s requerido) no cumple la versión mínima. Degradado.",
                    version,
                    self._min_version,
                )
                return
            self._model = artifact
            logger.info("Modelo v%s cargado desde %s", version, self._path)
        except Exception:  # noqa: BLE001
            logger.exception("Error al cargar %s", self._path)
            self._model = None

    @staticmethod
    def _lower(version: str) -> tuple[int, ...]:
        """Tupla numérica comparable (p.ej. '1.2.0' → (1, 2, 0))."""
        parts = []
        for chunk in version.split("."):
            digits = "".join(ch for ch in chunk if ch.isdigit())
            parts.append(int(digits) if digits else 0)
        return tuple(parts)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        return self._model


@lru_cache
def get_model_registry() -> ModelRegistry:
    """Singleton del registry del modelo."""
    settings: Settings = get_settings()
    return ModelRegistry(settings.model_path, settings.model_min_version)
