"""Orquestador del harness de integración IA ↔ .NET.

Levanta el microservicio IA real con uvicorn (proceso hijo), espera /health y
ejecuta el consumidor simulado .NET contra él. Al finalizar, apaga el proceso.

Uso:
    python scripts/integracion/harness.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
import m2m_keys  # noqa: E402

PORT = 8100
BASE_URL = f"http://127.0.0.1:{PORT}"


def _paso_validar_contrato() -> int:
    """Valida /openapi.json del servicio contra ia-api.yaml (openapi-spec-validator)."""
    print("PASO 0 — Validar OpenAPI emitido por el servicio contra ia-api.yaml")
    try:
        from openapi_spec_validator import validate_spec
    except ImportError:
        print("  [SKIP] openapi-spec-validator no instalado (requirements-dev).")
        return 0

    contrato_path = os.environ.get(
        "IA_CONTRATO_PATH",
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "medcare-contracts", "ia-api.yaml"
        ),
    )
    contrato = None
    if os.path.exists(contrato_path):
        import yaml

        with open(contrato_path, encoding="utf-8") as f:
            contrato = yaml.safe_load(f)
    else:
        print(f"  [SKIP] no se encontro ia-api.yaml en {contrato_path} (usa IA_CONTRATO_PATH).")
        return 0

    # 1) El spec que el servicio emite es OpenAPI válido.
    with urllib.request.urlopen(f"{BASE_URL}/openapi.json", timeout=10) as resp:
        emitido = json.load(resp)
    try:
        validate_spec(emitido)
        print("  [OK ] /openapi.json es un spec OpenAPI valido")
    except Exception as exc:
        print(f"  [FALLO] spec invalido: {exc}")
        return 1

    # 2) Toda ruta del contrato debe existir en el spec emitido (mismo path y método).
    no_cubiertas = []
    for path, methods in (contrato.get("paths") or {}).items():
        if path not in emitido["paths"]:
            no_cubiertas.append(f"{path} (sin path)")
            continue
        for method in methods:
            if method.lower() not in emitido["paths"][path]:
                no_cubiertas.append(f"{method.upper()} {path}")
    if no_cubiertas:
        print(f"  [FALLO] rutas del contrato no expuestas por el servicio: {no_cubiertas}")
        return 1
    print("  [OK ] todas las rutas de ia-api.yaml estan expuestas por /openapi.json")
    return 0


def _esperar_health(timeout: float = 45.0) -> bool:
    fin = time.time() + timeout
    while time.time() < fin:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/health", timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


def main() -> int:
    env = os.environ.copy()
    env.update(m2m_keys.env_para_servicio())
    env["IA_BASE_URL"] = BASE_URL

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(PORT),
        ],
        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print(f"Servicio IA arrancando (PID {proc.pid}) en {BASE_URL} ...")
    try:
        if not _esperar_health():
            print("FALLO: el servicio no respondio /health a tiempo.")
            return 1
        print("Servicio IA listo (modelo cargado).\n")
        if _paso_validar_contrato() != 0:
            return 1
        import importlib

        consumidor = importlib.import_module("consumidor_dotnet")
        consumidor.main()
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\nServicio IA detenido.")


if __name__ == "__main__":
    raise SystemExit(main())
