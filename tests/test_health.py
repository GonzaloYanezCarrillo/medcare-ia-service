"""Tests de GET /health (contrato ia-api.yaml)."""


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"healthy", "degraded", "unhealthy"}
    assert "version" in body
    assert isinstance(body["model_loaded"], bool)
