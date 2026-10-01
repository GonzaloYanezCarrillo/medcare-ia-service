"""Localización de `ia-api.yaml` (contrato de la API IA, owner: Dev C).

El contrato vive en el repo privado `medcare-contracts`. Este servicio usa:

1. `IA_CONTRATO_PATH` — ruta explícita (p. ej. el clon local del repo de contratos).
2. `<repo>/contracts/ia-api.yaml` — snapshot versionado en este repo (usado por CI).
3. `<repo>/../medcare-contracts/ia-api.yaml` — clon hermano en desarrollo local.

El snapshot permite que los tests de contrato corran en CI sin depender de un
repo privado. Debe resincronizarse cuando cambie el contrato original:

    Copy-Item <ruta-al-clon>/medcare-contracts/ia-api.yaml contracts/ia-api.yaml
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

NOMBRE_CONTRATO = "ia-api.yaml"


def candidatos_contrato() -> list[Path]:
    """Rutas candidatas en orden de prioridad."""
    lista: list[Path] = []
    if env_path := os.environ.get("IA_CONTRATO_PATH"):
        lista.append(Path(env_path))
    repo = Path(__file__).resolve().parents[1]
    lista.append(repo / "contracts" / NOMBRE_CONTRATO)
    lista.append(repo.parent / "medcare-contracts" / NOMBRE_CONTRATO)
    lista.append(Path(tempfile.gettempdir()) / "opencode" / "medcare-contracts" / NOMBRE_CONTRATO)
    return lista


def resolver_contrato() -> Path:
    """Devuelve la ruta del primer `ia-api.yaml` existente.

    Lanza FileNotFoundError si no hay ninguno (el snapshot versionado debería
    existir siempre en un clon del repo).
    """
    for candidato in candidatos_contrato():
        if candidato.exists():
            return candidato
    raise FileNotFoundError(
        f"No se encontro {NOMBRE_CONTRATO}. Rutas probadas: "
        + ", ".join(str(c) for c in candidatos_contrato())
    )
