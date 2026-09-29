"""Claves M2M para el harness de integración IA ↔ .NET (simula Supabase Auth).

El flujo real en producción: Supabase Auth emite el JWT RS256 para el cliente de
servicio `service-ia` (consumido por .NET), y el microservicio IA verifica la firma
contra el JWKS de Supabase. En el harness local no hay Supabase, así que se genera
un par RSA local: la pública se inyecta al servicio IA vía JWT_PUBLIC_KEY_PEM y el
consumidor simulado (rol .NET) firma los tokens con la privada.
"""

import datetime
import os
from datetime import timedelta

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

# Ruta donde se persiste el par RSA del harness (gitignore: scripts/integracion/.m2m/)
_KEYS_DIR = os.path.join(os.path.dirname(__file__), ".m2m")
_PRIVATE_PATH = os.path.join(_KEYS_DIR, "private.pem")
_PUBLIC_PATH = os.path.join(_KEYS_DIR, "public.pem")

ISSUER = os.environ.get("JWT_ISSUER", "https://<project-ref>.supabase.co/auth/v1")
AUDIENCE = os.environ.get("JWT_AUDIENCE", "authenticated")


def _generate_keypair() -> tuple[str, str]:
    """Genera un par RSA de 2048 (alg RS256) y devuelve (private_pem, public_pem)."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()
    public_pem = (
        private_key.public_key()
        .public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        .decode()
    )
    return private_pem, public_pem


def ensure_keypair() -> tuple[str, str]:
    """Devuelve el par RSA del harness, generándolo y persistido si no existe."""
    if os.path.exists(_PRIVATE_PATH) and os.path.exists(_PUBLIC_PATH):
        with open(_PRIVATE_PATH, encoding="utf-8") as f:
            private_pem = f.read()
        with open(_PUBLIC_PATH, encoding="utf-8") as f:
            public_pem = f.read()
        return private_pem, public_pem

    os.makedirs(_KEYS_DIR, exist_ok=True)
    private_pem, public_pem = _generate_keypair()
    with open(_PRIVATE_PATH, "w", encoding="utf-8") as f:
        f.write(private_pem)
    with open(_PUBLIC_PATH, "w", encoding="utf-8") as f:
        f.write(public_pem)
    return private_pem, public_pem


def public_pem_path() -> str:
    """Persiste el par si falta y devuelve la ruta al PEM público (para JWT_PUBLIC_KEY_PEM)."""
    ensure_keypair()
    return _PUBLIC_PATH


def make_token(*, role: str = "service-ia", exp_minutes: int = 30) -> str:
    """Emite un JWT RS256 como lo haría Supabase Auth para el servicio indicado."""
    private_pem, _ = ensure_keypair()
    now = datetime.datetime.now(datetime.UTC)
    payload = {
        "sub": "service-ia-consumer",
        "role": role,
        "iss": ISSUER,
        "aud": AUDIENCE,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=exp_minutes)).timestamp()),
    }
    return jwt.encode(payload, private_pem, algorithm="RS256")


def auth_headers(*, role: str = "service-ia") -> dict[str, str]:
    """Headers para las rutas protegidas, como los enviaría .NET."""
    return {"Authorization": f"Bearer {make_token(role=role)}"}


def env_para_servicio() -> dict[str, str]:
    """Variables de entorno que necesita el servicio IA para verificar los tokens del harness."""
    _, public_pem = ensure_keypair()
    return {
        "APP_ENV": "test",
        "JWT_ISSUER": ISSUER,
        "JWT_AUDIENCE": AUDIENCE,
        "JWT_PUBLIC_KEY_PEM": public_pem,
    }
