"""Tests de configuración (Settings).

Cubren la carga del PEM de clave pública desde archivo (`JWT_PUBLIC_KEY_PATH`),
el mecanismo pensado para los Secret Files de Render (montados en /etc/secrets/):
pegar un PEM multilínea a mano como env var es frágil (\\n literales o
truncamiento a la primera línea rompen el parseo), mientras que el archivo
secreto llega byte-exacto.
"""

import pytest

from app.config import Settings
from tests.auth_utils import PUBLIC_PEM


def test_settings_carga_pem_desde_ruta(tmp_path):
    """JWT_PUBLIC_KEY_PATH puebla jwt_public_key_pem con el contenido del archivo."""
    pem_file = tmp_path / "jwt_public.pem"
    pem_file.write_text(PUBLIC_PEM, encoding="utf-8")

    settings = Settings(jwt_public_key_path=str(pem_file))

    assert settings.jwt_public_key_pem == PUBLIC_PEM


def test_settings_sin_ruta_no_toca_el_pem():
    """Sin JWT_PUBLIC_KEY_PATH el comportamiento previo no cambia."""
    settings = Settings(jwt_public_key_path=None)

    assert settings.jwt_public_key_path is None


def test_settings_ruta_inexistente_falla_rapido(tmp_path):
    """Ruta definida pero archivo ausente: error claro al arrancar (fail-fast)."""
    with pytest.raises(ValueError, match="archivo inexistente"):
        Settings(jwt_public_key_path=str(tmp_path / "no_existe.pem"))


@pytest.mark.parametrize(
    "contenido",
    [
        # Modos de fallo reales de un PEM pegado como env var:
        # \n literales (dos caracteres) o valor truncado a la primera línea.
        pytest.param(PUBLIC_PEM.replace("\n", "\\n"), id="con_saltos_literal"),
        pytest.param(PUBLIC_PEM.split("\n")[0], id="solo_primera_linea"),
    ],
)
def test_settings_pem_deformado_falla_rapido(tmp_path, contenido):
    """PEM deformado en el archivo: falla al arrancar, no como 401 por request."""
    pem_file = tmp_path / "deformado.pem"
    pem_file.write_text(contenido, encoding="utf-8")

    with pytest.raises(ValueError, match="PEM inválido o truncado"):
        Settings(jwt_public_key_path=str(pem_file))
