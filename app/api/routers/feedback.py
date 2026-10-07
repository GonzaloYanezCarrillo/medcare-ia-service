"""Router de feedback de asistencia real (POST /feedback/asistencia, HU-IA-02 / C4-1)."""

from fastapi import APIRouter, Depends

from app.schemas.feedback import FeedbackAsistenciaRequest, FeedbackAsistenciaResponse
from app.services.auth import require_m2m
from app.services.feedback_store import FeedbackStore, get_feedback_store

router = APIRouter(tags=["Feedback"])

FB_AUTH = [Depends(require_m2m)]


@router.post(
    "/feedback/asistencia",
    response_model=FeedbackAsistenciaResponse,
    summary="Registrar desenlace real de una cita (feedback HU-IA-02)",
    dependencies=FB_AUTH,
)
def registrar_asistencia(
    request: FeedbackAsistenciaRequest,
    store: FeedbackStore = Depends(get_feedback_store),
) -> FeedbackAsistenciaResponse:
    """Persiste el desenlace + snapshot de features para el reentrenamiento (C4-1)."""
    registro, total = store.registrar(request.model_dump(mode="json"))
    return FeedbackAsistenciaResponse(
        cita_id=registro["cita_id"],
        asistio=bool(registro["asistio"]),
        registros_totales=total,
    )
