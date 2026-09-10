"""Registry del modelo entrenado.

Carga el artefacto `.joblib` desde `MODEL_PATH` (si existe) en el arranque, de modo que
`GET /health` y `POST /predict` reflejen su disponibilidad. Hasta que exista un modelo
entrenado (C1-1), el servicio opera en modo "degraded" con un predictor de marcado.
"""

import logging
from functools import lru_cache
from pathlib import Path

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Contenedor del identificador aprovisionado del modelo."""

    def __init__(self, model_path: str) -> None:
        self._path = Path(model_path)
        self._model = None
        self._load()

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning("Modelo no encontrado en %s. Servicio en modo degradado.", self._path)
            return
        try:
            import joblib

            self._model = joblib.load(self._path)
            logger.info("Modelo cargado desde %s", self._path)
        except Exception:  # noqa: BLE001
            logger.exception("Error al cargar %s", self._path)
            self._model = None

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
    return ModelRegistry(settings.model_path)
