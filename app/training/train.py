"""Entrenamiento del modelo base de no-show (C1-1, HU-IA-01).

Pipeline:
1. Cargar y limpiar el dataset Kaggle (ETL de `app.training.etl`).
2. Dividir train/test de forma estratificada (mantiene la proporción 20/80).
3. Transformar con ColumnTransformer: imputar con valor neutro las features que el
   dataset aún no expone (Especialidad, AusenciasPrevias, CanalRecordatorio → NaN),
   escalar numéricas y codificar categóricas (handle_unknown="ignore").
4. Entrenar RandomForest con `class_weight="balanced"` (desbalanceo 1:4).
5. Evaluar AUC-ROC, sensibilidad (recall minoritaria) y especificidad.
6. Persistir en `MODEL_PATH` un artefacto joblib con el pipeline y metadatos.

Cuando lleguen datos reales de producción con esas 3 columnas pobladas (C4-1+), se
reentrena con el mismo esquema y los imputers simplemente dejan de actuar.

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
from sklearn.impute import SimpleImputer
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
from app.training.etl import FEATURE_TYPES, FEATURES, load_cleaned

logger = logging.getLogger(__name__)

# Valor neutro con que se imputan las columnas ausentes (hoy todo NaN en el dataset).
CATEGORICAL_FILL = "desconocido"
NUMERIC_FILL = 0.0

# Modelo base C1-2: Weekday + RandomForest n=150, depth=12.
# vs baseline C1-1 (120/15 sin Weekday) mejora AUC (0.7049→0.7174) y sensibilidad
# (0.7188→0.7829) a costa de especificidad (0.5801→0.5490); para triaje de no-show
# priorizamos sensibilidad (detectar no-shows reales).
MODEL_KWARGS = {
    "n_estimators": 150,
    "max_depth": 12,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": 1,
}


def _imputer(value: Any) -> SimpleImputer:
    """SimpleImputer con valor constante (entrenar/inferir con columnas NaN)."""
    return SimpleImputer(strategy="constant", fill_value=value)


def build_pipeline(model_kwargs: dict | None = None) -> Pipeline:
    """Construye el pipeline: imputar → escalar/OHE → RandomForest balanced."""
    numeric_cols = [c for c in FEATURES if FEATURE_TYPES[c] == "numeric"]
    cat_cols = [c for c in FEATURES if FEATURE_TYPES[c] == "categorical"]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("imp", _imputer(NUMERIC_FILL)), ("scale", StandardScaler())]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", _imputer(CATEGORICAL_FILL)),
                        ("ohe", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(**(model_kwargs or MODEL_KWARGS))),
        ]
    )


def _defaults(x_train: pd.DataFrame) -> dict[str, Any]:
    """Valores neutros de entrenamiento para inferencia parcial.

    Mediana para numéricas, moda para categóricas; si la columna es todo NaN (los 3
    campos aún sin datos), se usa el mismo valor de imputación del pipeline.
    """
    defaults: dict[str, Any] = {}
    for col in x_train.columns:
        if FEATURE_TYPES[col] == "numeric":
            med = x_train[col].median()
            defaults[col] = float(NUMERIC_FILL) if pd.isna(med) else float(med)
        else:
            mode = x_train[col].mode()
            defaults[col] = CATEGORICAL_FILL if mode.empty else str(mode.iloc[0])
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
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
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
        "feature_types": FEATURE_TYPES,
        "metrics": metrics,
        "model_version": settings.app_version,
        "dataset_shape": {"train": int(len(x_train)), "test": int(len(x_test))},
        # Columnas del PredictRequest aún sin datos reales (imputadas a neutro).
        "missing_in_training": ["Especialidad", "AusenciasPrevias", "CanalRecordatorio"],
        # Valores neutros de entrenamiento para inferencia con PartialRequest.
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
