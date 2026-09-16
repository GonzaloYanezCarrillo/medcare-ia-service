"""Servicio de predicción de riesgo de no-show.

Usa el artefacto de entrenamiento (C1-1) cargado por `ModelRegistry`: un pipeline de
scikit-learn que espera un DataFrame con las features del dataset (`Age`, `Gender`,
`WaitingDays`, comorbilidades...). El contrato `PredictRequest` no expone todas esas
columnas, por lo que las no mapeadas se rellenan con los valores por defecto de
entrenamiento guardados en el artefacto (`defaults`).
"""

import logging

import pandas as pd

from app.schemas.predict import BandaRiesgo, PredictRequest, PredictResponse
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


def _score_to_banda(score: float) -> BandaRiesgo:
    """Mapea score (0-1) a banda según el umbral definido en el contrato."""
    if score < 0.25:
        return "Bajo"
    if score <= 0.50:
        return "Medio"
    return "Alto"


# Mapeo de campos del contrato PredictRequest → columnas del dataset de entrenamiento.
_REQUEST_TO_FEATURE = {
    "edad": "Age",
    "genero": "Gender",
    "dias_espera": "WaitingDays",
}


class PredictService:
    """Encapsula la lógica de inferencia de riesgo."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Calcula el score y la banda de riesgo para una cita."""
        score = self._infer_score(request)
        return PredictResponse(
            cita_id=request.cita_id,
            score_riesgo=round(score, 4),
            banda_riesgo=_score_to_banda(score),
        )

    def _infer_score(self, request: PredictRequest) -> float:
        if self._registry.is_loaded and self._registry.model is not None:
            return self._predict_with_model(request)

        # Modo degradado / de marcado (sin artefacto entrenado).
        return self._placeholder_score(request)

    def _predict_with_model(self, request: PredictRequest) -> float:
        artifact = self._registry.model
        frame = self._build_feature_frame(artifact, request)
        proba = artifact["pipeline"].predict_proba(frame)[0][1]
        return float(proba)

    @staticmethod
    def _build_feature_frame(artifact: dict, request: PredictRequest) -> pd.DataFrame:
        """Construye el DataFrame de features que espera el pipeline del artefacto."""
        row = dict(artifact["defaults"])
        for req_field, feature in _REQUEST_TO_FEATURE.items():
            row[feature] = getattr(request, req_field)
        return pd.DataFrame([row], columns=artifact["features"])

    @staticmethod
    def _placeholder_score(request: PredictRequest) -> float:
        # Heurística de marcado para operar sin artefacto (se reemplaza con el modelo real).
        base = min(request.dias_espera / 40.0 + request.ausencias_previas * 0.1, 1.0)
        if request.canal_recordatorio == "whatsapp":
            base = max(base - 0.05, 0.0)
        return max(min(base, 1.0), 0.0)
