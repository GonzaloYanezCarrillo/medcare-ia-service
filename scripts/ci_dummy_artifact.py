"""Genera un artefacto joblib sintético para validar el build de Dockerfile.prod en CI.

No es un modelo real: se entrena con datos sintéticos pequeños y constantes. Su único
propósito es que `COPY models/model.joblib` del Dockerfile.prod pueda ejecutarse y que el
smoke test de CI verifique que el contenedor responde `/health` en estado "healthy".

ADVERTENCIA: la versión del artefacto es válida (>= model_min_version) a propósito, para
que el flujo complete de CI sea verificable. Esta imagen se construye SOLO para smoke test
y éste workflow NO la publica en ningún registry: nunca debe desplegarse un artefacto
generado por este script en producción.

Ejecución (desde la raíz del repo, con PYTHONPATH=.):

    python scripts/ci_dummy_artifact.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from app.config import get_settings
from app.training.etl import FEATURE_TYPES, FEATURES
from sklearn.dummy import DummyClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent


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


if __name__ == "__main__":
    build_dummy_artifact(PROJECT_ROOT / get_settings().model_path)
    print("Artefacto sintético CI escrito en models/model.joblib")
