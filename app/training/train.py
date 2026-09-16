"""Entrenamiento del modelo base de no-show (C1-1, HU-IA-01).

Pipeline:
1. Cargar y limpiar el dataset Kaggle (ETL de `app.training.etl`).
2. Dividir train/test de forma estratificada (mantiene la proporción 20/80).
3. Transformar con ColumnTransformer (numéricas escaladas + binarias + categóricas).
4. Entrenar LogisticRegression con `class_weight="balanced"` (desbalanceo 1:4).
5. Evaluar AUC-ROC, sensibilidad (recall minoritaria) y especificidad.
6. Persistir en `MODEL_PATH` un artefacto joblib con el pipeline y metadatos.

Ejecución: `python -m app.training.train` desde la raíz del proyecto.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import Settings, get_settings
from app.training.etl import FEATURES, load_cleaned

logger = logging.getLogger(__name__)

CATEGORICAL = ["Gender", "Neighbourhood"]
BINARY = ["Scholarship", "Hipertension", "Diabetes", "Alcoholism", "Handcap"]
NUMERIC_WITH_SCALE = ["Age", "WaitingDays"]

# Features del request que el dataset no expone (no entrenables hoy): se documentan
# para la inferencia (se rellenan con valor neutro en `predict_service`).
REQUEST_ONLY = ["especialidad", "ausencias_previas", "canal_recordatorio"]

# Modelo base C1-1: RandomForest prioriza sensibilidad (detectar no-shows reales),
# con mejor AUC que LogisticRegression sobre el mismo split.
MODEL_KWARGS = {
    "n_estimators": 120,
    "max_depth": 15,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
}


def build_pipeline(model_kwargs: dict | None = None) -> Pipeline:
    """Construye el pipeline preprocesado + RandomForest."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_WITH_SCALE),
            ("bin", "passthrough", BINARY),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(**(model_kwargs or MODEL_KWARGS))),
        ]
    )


def _defaults(x_train: pd.DataFrame) -> dict[str, Any]:
    """Valores neutrales de entrenamiento para inferencia parcial.

    Mediana para numéricas, moda para categóricas, 0 para binarias.
    """
    defaults: dict[str, Any] = {}
    for col in x_train.columns:
        if col in NUMERIC_WITH_SCALE:
            defaults[col] = float(x_train[col].median())
        elif col in CATEGORICAL:
            defaults[col] = str(x_train[col].mode()[0])
        else:
            defaults[col] = 0
    return defaults


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Métricas de reporte: AUC, sensibilidad, especificidad, precisión, accuracy."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "auc_roc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "sensibilidad": round(float(tp / (tp + fn)), 4),
        "especificidad": round(float(tn / (tn + fp)), 4),
        "precision": round(float(precision_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "n_test": int(len(y_true)),
    }


def train(model_path: str | None = None, random_state: int = 42) -> dict[str, Any]:
    """Entrena el modelo base y persiste el artefacto.

    Returns:
        Dict con métricas y ruta del artefacto.
    """
    settings: Settings = get_settings()
    path = Path(model_path or settings.model_path)

    x, y = load_cleaned()
    # Subconjunto de features de entrenamiento.
    x_train, x_test, y_train, y_test = train_test_split(
        x[FEATURES], y, test_size=0.2, random_state=random_state, stratify=y
    )
    logger.info("Split: train=%d test=%d", len(x_train), len(x_test))

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test.values, y_pred, y_proba)

    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "binary": BINARY,
        "numeric_with_scale": NUMERIC_WITH_SCALE,
        "metrics": metrics,
        "model_version": settings.app_version,
        "dataset_shape": {"train": int(len(x_train)), "test": int(len(x_test))},
        # Valores por defecto de entrenamiento para inferencia parcial (el contrato
        # `PredictRequest` no expone Neighbourhood ni comorbilidades).
        "defaults": _defaults(x_train),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("Artefacto guardado en %s", path)
    logger.info("Métricas test: %s", json.dumps(metrics))
    return {"model_path": str(path), "metrics": metrics}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train()
