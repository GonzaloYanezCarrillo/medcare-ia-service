"""Servicio de predicción de riesgo de no-show.

Usa el artefacto de entrenamiento (C1-1) cargado por `ModelRegistry`. El pipeline espera
un DataFrame con las features del contrato `PredictRequest` (edad, género, días de
espera, especialidad, ausencias previas, canal de recordatorio) + `Weekday`. Las 3
primeras del contrato aún no tienen datos reales en el dataset de entrenamiento (quedan
sin señal hasta que haya datos de producción, C4-1+); en inferencia se rellenan desde
los `defaults` del artefacto.
"""

import logging

import pandas as pd

from app.schemas.predict import BandaRiesgo, ClasePrediccion, PredictRequest, PredictResponse
from app.services.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

# Umbral de decisión para la clase discreta: score >= 0.5 → no_asiste.
_CLASE_THRESHOLD = 0.5


def _score_to_banda(score: float) -> BandaRiesgo:
    """Mapea score (0-1) a banda según el umbral definido en el contrato."""
    if score < 0.25:
        return "Bajo"
    if score <= 0.50:
        return "Medio"
    return "Alto"


def _score_to_clase(score: float) -> ClasePrediccion:
    """Predicción discreta (asiste/no_asiste) según el umbral de decisión."""
    return "no_asiste" if score >= _CLASE_THRESHOLD else "asiste"


# Mapeo de campos del contrato PredictRequest → features del modelo (biunívoco).
_REQUEST_TO_FEATURE = {
    "edad": "Age",
    "genero": "Gender",
    "dias_espera": "WaitingDays",
    "especialidad": "Especialidad",
    "ausencias_previas": "AusenciasPrevias",
    "canal_recordatorio": "CanalRecordatorio",
}


class PredictService:
    """Encapsula la lógica de inferencia de riesgo."""

    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry

    def predict(self, request: PredictRequest) -> PredictResponse:
        """Calcula el score, la banda y la clase de riesgo para una cita."""
        score = self._infer_score(request)
        return PredictResponse(
            cita_id=request.cita_id,
            score_riesgo=round(score, 4),
            banda_riesgo=_score_to_banda(score),
            clase=_score_to_clase(score),
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
        frame = pd.DataFrame([row], columns=artifact["features"])
        # Coercer dtypes a los tipos con que se entrenó el pipeline: numéricas a
        # float64, categóricas a str (compatibilidad request ↔ SimpleImputer).
        feature_types = artifact.get("feature_types", {})
        for col in artifact["features"]:
            if feature_types.get(col) == "categorical":
                frame[col] = frame[col].astype(str)
            elif feature_types.get(col) == "numeric":
                frame[col] = frame[col].astype("float64")
        return frame

    @staticmethod
    def _placeholder_score(request: PredictRequest) -> float:
        # Heurística de marcado para operar sin artefacto (se reemplaza con el modelo real).
        base = min(request.dias_espera / 40.0 + request.ausencias_previas * 0.1, 1.0)
        if request.canal_recordatorio == "whatsapp":
            base = max(base - 0.05, 0.0)
        return max(min(base, 1.0), 0.0)
