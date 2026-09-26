"""Fixtures compartidos para los tests."""

import pytest

# Expone la clave pública de prueba ANTES de importar app.main: ese módulo ejecuta
# create_app() → get_settings() (lru_cache) al importarse, y el verifier la cachea.
from tests.auth_utils import auth_headers, set_jwt_env

set_jwt_env()

from app.main import create_app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture()
def client():
    return TestClient(create_app())


@pytest.fixture()
def headers() -> dict[str, str]:
    return auth_headers()
