"""Fixtures compartidos para los tests."""

import pytest

# Expone la clave pública de prueba ANTES de importar app.main: ese módulo ejecuta
# create_app() → get_settings() (lru_cache) al importarse, y el verifier la cachea.
from tests.auth_utils import auth_headers, set_jwt_env

set_jwt_env()

from app.main import create_app  # noqa: E402
from app.services.feedback_store import FeedbackStore, get_feedback_store  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def feedback_store(tmp_path) -> FeedbackStore:
    """Store de feedback en tmp: los tests nunca escriben en data/ real."""
    return FeedbackStore(tmp_path / "asistencia.jsonl")


@pytest.fixture()
def client(feedback_store: FeedbackStore):
    app = create_app()
    app.dependency_overrides[get_feedback_store] = lambda: feedback_store
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def headers() -> dict[str, str]:
    return auth_headers()
