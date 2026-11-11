"""Configuración del microservicio de IA.

Carga valores desde variables de entorno y un archivo `.env` (via pydantic-settings).
Coherente con `contracts/ia-api.yaml`.
"""

from functools import lru_cache
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Parámetros de configuración del servicio IA."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "MedCare IA — API Microservicio IA"
    app_version: str = "1.4.0"
    app_env: str = "development"

    api_prefix: str = ""

    # Infraestructura del modelo
    model_path: str = "models/model.joblib"
    model_min_version: str = "0.1.0"

    # Dataset Kaggle (C0-3 / C1-1)
    dataset_path: str = "data/KaggleV2-May-2016.csv"

    # Feedback de asistencia real (C4-1 / HU-IA-02): JSONL append-upsert por cita_id.
    # Solo features + etiqueta (RNF-SEG-03: sin contenido clínico).
    feedback_path: str = "data/feedback/asistencia.jsonl"

    # Integración con la API Core .NET (Sprint 2+)
    core_api_url: str | None = None
    core_token: str | None = None

    # Auth M2M (C5-1/HU-AUTH-03): validación de JWT que recibe de .NET.
    # El contrato auth.yaml define: JWT RS256 emitido por Supabase Auth.
    # En producción se resuelve la clave pública vía JWKS_URL (well-known de Supabase).
    # En dev/test se inyecta JWT_PUBLIC_KEY_PEM para tokens autocontenidos (sin Supabase).
    # Alternativa robusta (Render): JWT_PUBLIC_KEY_PATH al Secret File montado en
    # /etc/secrets/ — los env vars de campo único colapsan los PEM multilínea.
    jwt_jwks_url: str | None = None
    jwt_public_key_pem: str | None = None
    jwt_public_key_path: str | None = None
    jwt_issuer: str = "https://weqohesdsgxepjilsgae.supabase.co/auth/v1"
    jwt_audience: str = "authenticated"
    jwt_required_role: str = "service-ia"

    @field_validator("api_prefix")
    @classmethod
    def _normalize_prefix(cls, value: str) -> str:
        value = (value or "").strip()
        if value and not value.startswith("/"):
            value = f"/{value}"
        return value.rstrip("/")

    @model_validator(mode="after")
    def _load_pem_from_path(self) -> "Settings":
        """Puebla `jwt_public_key_pem` desde `jwt_public_key_path`.

        Los paneles de env vars (p.ej. Render) suelen colapsar los saltos de línea
        de un PEM multilínea pegado a mano: la firma deja de validar y toda llamada
        regresa 401. Como Secret File (montado en /etc/secrets/) el contenido llega
        byte-exacto. Fail-fast: si la ruta está definida pero el archivo falta o el
        PEM es inválido, el arranque falla con error claro en los logs en vez de
        rechazar tokens silenciosamente en cada request.
        """
        if self.jwt_public_key_path:
            path = Path(self.jwt_public_key_path)
            if not path.is_file():
                raise ValueError(
                    f"JWT_PUBLIC_KEY_PATH={path}: archivo inexistente. "
                    "¿Está montado el Secret File en /etc/secrets/?"
                )
            pem = path.read_text(encoding="utf-8")
            try:
                serialization.load_pem_public_key(pem.encode())
            except ValueError as exc:
                raise ValueError(
                    f"JWT_PUBLIC_KEY_PATH={path}: PEM inválido o truncado "
                    f"(¿se perdieron los saltos de línea?): {exc}"
                ) from exc
            self.jwt_public_key_pem = pem
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env.lower() == "production"


@lru_cache
def get_settings() -> Settings:
    """Devuelve la configuración (cacheada)."""
    return Settings()
