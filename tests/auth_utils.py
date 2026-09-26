"""Utilidades de auth para tests: par RSA de prueba y emisión de JWTs.

Permite probar las rutas protegidas (C5-1) sin depender de Supabase: los tests
firman tokens con la privada local y `JWT_PUBLIC_KEY_PEM` expone la pública que
`app.services.auth.get_auth_verifier` usará para verificarlos.
"""

import datetime
import os
from datetime import timedelta

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

ISSUER = os.environ.get("JWT_ISSUER", "https://<project-ref>.supabase.co/auth/v1")
AUDIENCE = os.environ.get("JWT_AUDIENCE", "authenticated")

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_PEM = _PRIVATE_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()
PUBLIC_PEM = (
    _PRIVATE_KEY.public_key()
    .public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode()
)

# Segunda clave: tokens firmados con ella fallan la verificación (firma incorrecta).
_FOREIGN_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_FOREIGN_PEM = _FOREIGN_KEY.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.PKCS8,
    encryption_algorithm=serialization.NoEncryption(),
).decode()


def make_token(
    *,
    role: str = "service-ia",
    exp_minutes: int | None = 30,
    issuer: str = ISSUER,
    audience: str = AUDIENCE,
    foreign_key: bool = False,
) -> str:
    """Emite un JWT RS256 de prueba.

    `foreign_key=True` firma con una clave distinta (para probar la firma inválida).
    `exp_minutes=None` omite `exp` (token sin expiración).
    """
    now = datetime.datetime.now(datetime.UTC)
    header = {"alg": "RS256", "typ": "JWT"}
    payload = {
        "sub": "microservice-test",
        "role": role,
        "iss": issuer,
        "aud": audience,
        "iat": int(now.timestamp()),
    }
    if exp_minutes is not None:
        payload["exp"] = int((now + timedelta(minutes=exp_minutes)).timestamp())
    key = _FOREIGN_PEM if foreign_key else _PRIVATE_PEM
    return jwt.encode(payload, key, algorithm="RS256", headers=header)


def auth_headers(*, role: str = "service-ia") -> dict[str, str]:
    """Headers con token válido para las rutas protegidas."""
    return {"Authorization": f"Bearer {make_token(role=role)}"}


def set_jwt_env() -> None:
    """Expone la clave pública de prueba para que el verifier la use."""
    os.environ["JWT_PUBLIC_KEY_PEM"] = PUBLIC_PEM
    os.environ.setdefault("JWT_ISSUER", ISSUER)
    os.environ.setdefault("JWT_AUDIENCE", AUDIENCE)
