"""Fixtures compartidos para los tests."""

import pytest
from app.main import create_app
from fastapi.testclient import TestClient


@pytest.fixture()
def client():
    return TestClient(create_app())
