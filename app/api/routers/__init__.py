"""Capa de routers HTTP."""

from app.api.routers.health import router as health_router
from app.api.routers.nlp import router as nlp_router
from app.api.routers.predict import router as predict_router

all_routers = [health_router, nlp_router, predict_router]

__all__ = ["all_routers"]
