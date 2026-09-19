"""Punto de entrada del microservicio de IA.

Levanta la aplicación FastAPI con los routers definidos en `app/api/routers`.
Documentación interactiva en /docs (Swagger) y /redoc.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routers import all_routers
from app.config import Settings, get_settings
from app.services.model_registry import get_model_registry

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = get_settings()
    logger.info("Iniciando %s (env=%s)", settings.app_name, settings.app_env)
    # Pre-cargar el modelo en startup: el primer /predict y /health ya lo encuentran
    # en memoria y reflejan su disponibilidad real desde el arranque.
    registry = await asyncio.to_thread(get_model_registry)
    if registry.is_loaded:
        logger.info("Modelo cargado en el arranque.")
    else:
        logger.warning("Modelo no disponible: servicio en modo degradado.")
    yield


def create_app() -> FastAPI:
    settings: Settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description=(
            "Microservicio de IA de MedCare. Owner: Dev C (Datos e IA). "
            "Stack: FastAPI · Scikit-Learn · spaCy/NLTK · joblib."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    prefix = settings.api_prefix or ""
    for router in all_routers:
        app.include_router(router, prefix=prefix)

    return app


app = create_app()
