"""Autenticación M2M (C5-1/HU-AUTH-03).

Verifica que el `Authorization: Bearer <JWT>` que recibe .NET sea válido según el
contrato `auth.yaml`:

- Firma: RS256 (clave pública resuelta por JWKS de Supabase en producción, o una
  clave PEM inyectada en desarrollo/pruebas para no depender de Supabase).
- Claims: `iss` y `aud` del emisor, `exp` no vencido.
- Autorización: `role == service-ia` (el único rol permitido por ia-api.yaml).

Diseño de seguridad (fail-closed):
- Sin `Authorization` → 401.
- Firma/claims inválidos o vencidos → 401.
- Rol distinto de `service-ia` → 403 (autenticado pero no autorizado).
- Sin JWKS_URL ni public key PEM configurados → 503 (el servicio niega el tráfico
  en lugar de abrir las rutas: un despliegue sin auth es una falla de infraestructura).
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import jwt
from fastapi import Depends, Header, HTTPException, status
from jwt import PyJWKClient

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Solo RS256 (contrato auth.yaml, algoritmo 9). Rechazar algoritmos débiles (HS*/none).
_ALLOWED_ALGORITHMS = ["RS256"]
_BEARER_PREFIX = "Bearer "
_WWW_AUTHENTICATE = {"WWW-Authenticate": "Bearer"}


class JwtVerifier:
    """Valida JWTs M2M contra una clave pública (JWKS remoto o PEM local)."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._jwks_client = PyJWKClient(settings.jwt_jwks_url) if settings.jwt_jwks_url else None

    def _key(self, token: str) -> Any:
        """Resuelve la clave pública del emisor.

        Prefiere la clave PEM inyectada (dev/test deterministas); si no hay, usa el
        JWKS remoto de Supabase (producción). Sin ninguna de las dos → 503.
        """
        if self._settings.jwt_public_key_pem:
            return self._settings.jwt_public_key_pem
        if self._jwks_client is not None:
            return self._jwks_client.get_signing_key_from_jwt(token).key
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Auth M2M no configurado: falta JWT_JWKS_URL o JWT_PUBLIC_KEY_PEM. "
                "El servicio niega el tráfico hasta definir una fuente de claves."
            ),
        )

    def verify(self, token: str) -> dict[str, Any]:
        """Verifica firma + claims y devuelve el payload si `role == service-ia`."""
        try:
            key = self._key(token)
            payload = jwt.decode(
                token,
                key,
                algorithms=_ALLOWED_ALGORITHMS,
                issuer=self._settings.jwt_issuer,
                audience=self._settings.jwt_audience,
                options={"require": ["exp", "iss", "aud"]},
            )
        except jwt.ExpiredSignatureError as exc:
            logger.info("Token M2M vencido.")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expirado.",
                headers=_WWW_AUTHENTICATE,
            ) from exc
        except jwt.InvalidTokenError as exc:
            logger.info("Token M2M inválido: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token inválido o firma incorrecta.",
                headers=_WWW_AUTHENTICATE,
            ) from exc

        role = payload.get("role")
        if role != self._settings.jwt_required_role:
            logger.info("Token M2M con rol no autorizado: %r", role)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Rol `{self._settings.jwt_required_role}` requerido.",
            )
        return payload


@lru_cache
def get_auth_verifier() -> JwtVerifier:
    """Singleton del verificador M2M (mismo patrón que get_model_registry)."""
    return JwtVerifier(get_settings())


def _extract_token(authorization: str | None) -> str:
    """Valida el header y extrae el token JWT; sin header/vacío → 401."""
    if not authorization or not authorization.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Encabezado `Authorization: Bearer <JWT>` requerido.",
            headers=_WWW_AUTHENTICATE,
        )
    return authorization[len(_BEARER_PREFIX) :].strip()


def require_m2m(
    authorization: str | None = Header(default=None),
    verifier: JwtVerifier = Depends(get_auth_verifier),
) -> dict[str, Any]:
    """Dependencia FastAPI: autentica y autoriza llamadas machine-to-machine.

    Se inyecta en las rutas protegidas (x-roles: [service-ia] en ia-api.yaml).
    """
    return verifier.verify(_extract_token(authorization))
