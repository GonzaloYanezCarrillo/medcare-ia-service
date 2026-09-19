"""Tests del ciclo de vida de la app (C2-1): pre-carga del modelo en startup."""

import app.main as main
from app.api.routers import health as health_router
from app.main import create_app
from fastapi.testclient import TestClient


def _patch_registry(monkeypatch, registry):
    """Parchea el registry en los puntos de entrada: lifespan y router /health."""
    from app.services import model_registry

    monkeypatch.setattr(model_registry, "get_model_registry", lambda: registry)
    monkeypatch.setattr(main, "get_model_registry", lambda: registry)
    monkeypatch.setattr(health_router, "get_model_registry", lambda: registry)


def test_modelo_se_precarga_en_startup(monkeypatch):
    """El lifespan debe invocar `get_model_registry` antes de servir requests."""

    class FakeRegistry:
        is_loaded = True
        model = {"pipeline": object()}

    _patch_registry(monkeypatch, FakeRegistry())

    with TestClient(create_app()) as client:
        # El TestClient con context manager ejecuta el lifespan al entrar.
        assert client.get("/health").status_code == 200


def test_modelo_ausente_queda_degradado(monkeypatch):
    """Sin artefacto, el arranque no lanza excepción y /health reporta degraded."""

    class EmptyRegistry:
        is_loaded = False
        model = None

    _patch_registry(monkeypatch, EmptyRegistry())

    with TestClient(create_app()) as client:
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["model_loaded"] is False
