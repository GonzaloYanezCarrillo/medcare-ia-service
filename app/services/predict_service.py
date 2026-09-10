"""Servicio de predicción de riesgo de no-show.

Hasta que exista un modelo real (C1-1), devuelve un score determinístico de marcado que
permite validar el contrato y el flujo end-to-end. Se reemplaza por la inferencia de
Scikit-Learn cuando el artefacto esté disponible.
"""

from app.schemas.predict import BandaRiesgo, PredictRequest, PredictResponse
from app.services.model_registry import ModelRegistry


def _score_to_banda(score: float) -> BandaRiesgo:
    """Mapea score (0-1) a banda según el umbral definido en el contrato."""
    if score < 0.25:
        return "Bajo"
    if score <= 0.50:
        return "Medio"
    return "Alto"


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

        # Modo degradado / de marcado (placeholder hasta C1-1).
        return self._placeholder_score(request)

    def _predict_with_model(self, request: PredictRequest) -> float:
        model = self._registry.model
        features = self._feature_vector(request)
        proba = model.predict_proba(features)[0][1]
        return float(proba)

    @staticmethod
    def _feature_vector(request: PredictRequest):
        import numpy as np

        return np.array(
            [
                request.edad,
                request.dias_espera,
                request.ausencias_previas,
                1.0 if request.genero == "F" else 0.0,
            ]
        ).reshape(1, -1)

    @staticmethod
    def _placeholder_score(request: PredictRequest) -> float:
        # Heurística de marcado para validar el contrato (se elimina con el modelo real).
        base = min(request.dias_espera / 40.0 + request.ausencias_previas * 0.1, 1.0)
        if request.canal_recordatorio == "whatsapp":
            base = max(base - 0.05, 0.0)
        return max(min(base, 1.0), 0.0)
