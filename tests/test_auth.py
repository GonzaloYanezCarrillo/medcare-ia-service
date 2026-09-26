"""Tests de Auth M2M (C5-1/HU-AUTH-03).

Valida el fail-closed del contrato ia-api.yaml: /predict y /nlp/* exigen
`Authorization: Bearer <JWT>` con rol `service-ia`; /health queda público.
"""

from tests.auth_utils import auth_headers, make_token


def _predict(client, headers):
    import uuid

    cita_id = str(uuid.uuid4())
    return client.post(
        "/predict",
        headers=headers,
        json={
            "cita_id": cita_id,
            "edad": 34,
            "genero": "F",
            "dias_espera": 5,
            "especialidad": "psicologia",
            "ausencias_previas": 1,
            "canal_recordatorio": "whatsapp",
        },
    )


def test_predict_exige_token_401(client):
    response = _predict(client, headers={})
    assert response.status_code == 401
    assert response.json()["detail"]


def test_predict_rechaza_token_invalido_401(client):
    response = _predict(client, headers={"Authorization": "Bearer no-es-jwt"})
    assert response.status_code == 401


def test_predict_rechaza_firma_incorrecta_401(client):
    headers = {"Authorization": f"Bearer {make_token(foreign_key=True)}"}
    response = _predict(client, headers=headers)
    assert response.status_code == 401


def test_predict_rechaza_token_vencido_401(client):
    headers = {"Authorization": f"Bearer {make_token(exp_minutes=-30)}"}
    response = _predict(client, headers=headers)
    assert response.status_code == 401


def test_predict_rechaza_rol_distinto_403(client):
    headers = {"Authorization": f"Bearer {make_token(role='service-omro')}"}
    response = _predict(client, headers=headers)
    assert response.status_code == 403


def test_predict_acepta_token_service_ia_200(client):
    response = _predict(client, headers=auth_headers())
    assert response.status_code == 200


def test_nlp_exige_token_401(client):
    response = client.post("/nlp/sintomas", headers={}, json={"texto_sintomas": "dolor"})
    assert response.status_code == 401


def test_nlp_acepta_token_service_ia_200(client):
    response = client.post(
        "/nlp/sintomas", headers=auth_headers(), json={"texto_sintomas": "dolor"}
    )
    assert response.status_code == 200


def test_health_es_publico_sin_token_200(client):
    response = client.get("/health")
    assert response.status_code == 200
