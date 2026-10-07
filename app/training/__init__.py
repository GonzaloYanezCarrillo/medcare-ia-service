"""Módulo de entrenamiento del modelo de riesgo de no-show (Track C).

Responsabilidades:
- C0-3: selección y descarga del dataset Kaggle (*Medical Appointment No Shows*).
- C1-1: ETL, ingeniería de features, entrenamiento del modelo base y artefacto `.joblib`.
- C4-1: reentrenamiento con feedback real (HU-IA-02) y versionado.
"""

from app.training.etl import FEATURES, clean, load_cleaned, load_feedback
from app.training.train import build_pipeline, evaluar_contra_baseline, siguiente_version, train

__all__ = [
    "FEATURES",
    "clean",
    "evaluar_contra_baseline",
    "load_cleaned",
    "load_feedback",
    "build_pipeline",
    "siguiente_version",
    "train",
]
