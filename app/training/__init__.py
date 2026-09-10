"""Módulo de entrenamiento del modelo de riesgo de no-show (Track C).

Responsabilidades planificadas:
- C0-3: selección y descarga del dataset Kaggle (*Medical Appointment No Shows*).
- C1-1: ETL, ingeniería de features, entrenamiento del modelo base y artefacto `.joblib`.
- C4-1: reentrenamiento con feedback real (HU-IA-02) y versionado.
"""

from app.training.features import build_feature_vector
from app.training.train import train_placeholder

__all__ = ["build_feature_vector", "train_placeholder"]
