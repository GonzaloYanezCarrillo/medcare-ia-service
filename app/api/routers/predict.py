"""Routers de predicción de riesgo (POST /predict)."""

from fastapi import APIRouter, Depends

from app.schemas.predict import PredictRequest, PredictResponse
from app.services.model_registry import get_model_registry
from app.services.predict_service import PredictService

router = APIRouter(tags=["ML"])


def get_predict_service() -> PredictService:
    """Provider del servicio de predicción (inyección manual)."""
    return PredictService(get_model_registry())


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predicción de riesgo de no-show bajo demanda",
)
def predecir_riesgo(
    request: PredictRequest, service: PredictService = Depends(get_predict_service)
) -> PredictResponse:
    """Calcula el riesgo de no-show (score + banda) para una cita."""
    return service.predict(request)
