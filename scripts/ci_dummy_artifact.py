"""Genera un artefacto joblib sintético para validar el build de Dockerfile.prod en CI.

No es un modelo real: se entrena con datos sintéticos pequeños y constantes. Su único
propósito es que `COPY models/model.joblib` del Dockerfile.prod pueda ejecutarse y que el
smoke test de CI verifique que el contenedor responde `/health` en estado "healthy".

ADVERTENCIA: la versión del artefacto es válida (>= model_min_version) a propósito, para
que el flujo complete de CI sea verificable. Esta imagen se construye SOLO para smoke test
y éste workflow NO la publica en ningún registry: nunca debe desplegarse un artefacto
generado por este script en producción.

Ejecución (desde cualquier directorio, sin PYTHONPATH necesario):

    python scripts/ci_dummy_artifact.py

Si en el path de salida ya existe un artefacto con métricas reales (AUC > 0), el script
aborta para no pisar un modelo de producción. En CI el workspace está limpio, no hay nada
que sobrescribir; si se desea forzar (p. ej. regenerar el dummy a propósito), usar `--force`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

# Permitir `import app.*` sin depender del PYTHONPATH del ejecutor (CI/local).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.training.etl import FEATURE_TYPES, FEATURES  # noqa: E402


class DummyPipeline:
    """Pipeline mínimo compatible con `predict_proba` para el artefacto sintético."""

    def __init__(self, n_classes: int = 2) -> None:
        self._clf = DummyClassifier(strategy="prior", random_state=0)
        self.n_features_in_ = len(FEATURES)
        self.classes_ = np.arange(n_classes, dtype=int)

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> DummyPipeline:
        self._clf.fit(x, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        return self._clf.predict_proba(x)


def build_dummy_artifact(out_path: Path) -> None:
    """Entrena un dummy y persiste el artefacto con la estructura de C1.1."""
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "Age": rng.uniform(0, 90, n),
            "WaitingDays": rng.uniform(0, 60, n),
            "Weekday": rng.integers(0, 7, n),
            "Gender": rng.choice(["M", "F"], n),
            "Especialidad": rng.choice(["A", "B"], n),
            "AusenciasPrevias": rng.integers(0, 5, n),
            "CanalRecordatorio": rng.choice(["email", "whatsapp", "ninguno"], n),
        }
    )
    y = rng.integers(0, 2, n)
    pipeline = DummyPipeline().fit(frame, y)

    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "feature_types": FEATURE_TYPES,
        "metrics": {"auc_roc": 0.0, "sensibilidad": 0.0, "especificidad": 0.0},
        "model_version": get_settings().app_version,
        "dataset_shape": {"train": n, "test": 0},
        "missing_in_training": [],
        "defaults": {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_path)


def ensure_safe_overwrite(out_path: Path, force: bool) -> None:
    """Aborta si en `out_path` ya hay un artefacto con métricas reales (AUC > 0).

    El dummy se genera en CI (workspace limpio), pero donde corre manualmente nunca debe
    pisar un modelo entrenado sin que se pida explícitamente con `--force`.
    """
    if force or not out_path.exists():
        return
    try:
        existing = joblib.load(out_path)
    except Exception:
        return
    auc = (existing.get("metrics") or {}).get("auc_roc", 0) if isinstance(existing, dict) else 0
    if auc > 0:
        raise SystemExit(
            f"{out_path} contiene un modelo real (auc_roc={auc}). "
            "Abortando para no pisar el artefacto de producción. Use --force si es intencional."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="sobrescribir un artefacto real existente"
    )
    args = parser.parse_args()

    out_path = PROJECT_ROOT / get_settings().model_path
    ensure_safe_overwrite(out_path, args.force)
    build_dummy_artifact(out_path)
    print(f"Artefacto sintético CI escrito en {out_path.name}")
