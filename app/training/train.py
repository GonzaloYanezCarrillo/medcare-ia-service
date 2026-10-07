"""Entrenamiento del modelo base de no-show (C1-1, HU-IA-01).

Pipeline:
1. Cargar y limpiar el dataset Kaggle (ETL de `app.training.etl`).
2. Dividir train/test de forma estratificada (mantiene la proporción 20/80).
3. Transformar con ColumnTransformer: escalar numéricas y codificar categóricas
   (handle_unknown="ignore"). Las features que el dataset aún no expone
   (Especialidad, AusenciasPrevias, CanalRecordatorio → NaN) quedan sin señal hasta
   que haya datos reales.
4. Entrenar RandomForest con `class_weight="balanced"` (desbalanceo 1:4).
5. Evaluar AUC-ROC, sensibilidad (recall minoritaria) y especificidad.
6. Persistir en `MODEL_PATH` un artefacto joblib con el pipeline y metadatos;
   versión semver del modelo (independiente de `app_version`) y archivo en
   `models/versions/<semver>/` + puntero `models/latest.json` (C4-1).

Cuando el modelo se reentrene con datos reales de producción (C4-1+) que pueblen esas
columnas, empezarán a aportar al modelo con el mismo esquema — sin cambios de código.

Ejecución: `python -m app.training.train` desde la raíz del proyecto.
"""

from __future__ import annotations

import json
import logging
import shutil
from datetime import UTC, datetime
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
from app.training.etl import FEATURE_TYPES, FEATURES, load_cleaned, load_feedback

logger = logging.getLogger(__name__)

# Tolerancia del gate de no-degradación (C4-1): el modelo nuevo solo se promueve
# si AUC-ROC y sensibilidad no caen por debajo de la línea base más allá de epsilon.
GATE_EPSILON = 0.01

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


def evaluar_contra_baseline(
    nuevas: dict[str, float], baseline: dict[str, float], epsilon: float = GATE_EPSILON
) -> tuple[bool, list[str]]:
    """Gate de no-degradación del reentrenamiento (C4-1).

    El modelo nuevo solo se promueve si AUC-ROC y sensibilidad no caen más allá
    de `epsilon` por debajo de la línea base (prioridad: detectar no-shows).
    Devuelve (promovido, motivos_de_rechazo).
    """
    motivos: list[str] = []
    for metrica in ("auc_roc", "sensibilidad"):
        base = baseline.get(metrica)
        nueva = nuevas.get(metrica)
        if base is None or nueva is None:
            continue
        if nueva < base - epsilon:
            motivos.append(f"{metrica}: {nueva} < baseline {base} - {epsilon}")
    return (not motivos, motivos)


def _artefacto_actual(path: Path) -> dict[str, Any] | None:
    """Artefacto vigente en `path` (línea base del gate), si existe y es legible."""
    if not path.exists():
        return None
    try:
        artifact = joblib.load(path)
    except Exception:  # noqa: BLE001
        logger.exception("No se pudo leer el artefacto vigente en %s", path)
        return None
    return artifact if isinstance(artifact, dict) else None


def siguiente_version(version_actual: str | None, con_feedback: bool) -> str:
    """Siguiente versión semver del MODELO, independiente de `app_version`.

    Regla (C4-1): sin artefacto previo → `1.0.0`; reentrenamiento con feedback
    real (datos nuevos) → bump menor; reentrenamiento del mismo dataset →
    bump patch. Un cambio rompiente del set de features sería bump mayor.
    """
    if not version_actual:
        return "1.0.0"
    partes = version_actual.split(".")
    if len(partes) != 3 or not all(p.isdigit() for p in partes):
        logger.warning("Versión previa inválida %r; se reinicia a 1.0.0", version_actual)
        return "1.0.0"
    mayor, menor, parche = (int(p) for p in partes)
    if con_feedback:
        return f"{mayor}.{menor + 1}.0"
    return f"{mayor}.{menor}.{parche + 1}"


def _archivar_version(path: Path, artifact: dict[str, Any]) -> None:
    """Copia el artefacto promovido a `versions/<v>/` y actualiza `latest.json` (C4-1).

    `model_path` sigue siendo el artefacto activo (runtime sin cambios); el
    archivo `versions/` conserva el linaje y `latest.json` el puntero auditable.
    """
    version = str(artifact["model_version"])
    destino = path.parent / "versions" / version / path.name
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destino)
    latest = {
        "model_version": version,
        "parent_version": artifact.get("parent_version"),
        "trained_at": artifact.get("trained_at"),
        "path": destino.relative_to(path.parent).as_posix(),
        "metrics": artifact.get("metrics"),
        "training_data": artifact.get("training_data"),
    }
    (path.parent / "latest.json").write_text(
        json.dumps(latest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("Versión %s archivada en %s", version, destino)


def _delta_metrics(
    nuevas: dict[str, Any], baseline: dict[str, Any] | None
) -> dict[str, float] | None:
    """Delta retrained − baseline de las métricas numéricas comunes (reporte C4-1)."""
    if not baseline:
        return None
    return {
        clave: round(float(nuevas[clave]) - float(baseline[clave]), 6)
        for clave in nuevas
        if clave in baseline
        and isinstance(nuevas[clave], float)
        and isinstance(baseline[clave], float)
    }


def _registrar_comparativo(path: Path, payload: dict[str, Any]) -> None:
    """Escribe `metrics.json` junto al artefacto: baseline vs. reentrenado (C4-1)."""
    destino = path.parent / "metrics.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Comparativo de métricas en %s", destino)


def train(
    model_path: str | None = None,
    random_state: int = 42,
    include_feedback: bool = False,
    epsilon: float = GATE_EPSILON,
) -> dict[str, Any]:
    """Entrena el modelo base y persiste el artefacto.

    Con `include_feedback=True` (C4-1 / HU-IA-02) concatena el feedback de
    asistencia real al dataset Kaggle y aplica el **gate de no-degradación**:
    si las métricas degradan la línea base más allá de `epsilon`, NO se
    sobrescribe el artefacto (`promovido=False`) y el vigente sigue sirviendo.

    Returns:
        Dict con métricas, ruta, `promovido` y `motivos` (vacío si se promovió).
    """
    settings: Settings = get_settings()
    path = Path(model_path or settings.model_path)

    x, y = load_cleaned()
    kaggle_rows = int(len(x))
    feedback_rows = 0
    if include_feedback:
        x_fb, y_fb = load_feedback()
        feedback_rows = int(len(x_fb))
        if feedback_rows:
            x = pd.concat([x, x_fb], ignore_index=True)
            y = pd.concat([y, y_fb], ignore_index=True)
        logger.info("Feedback HU-IA-02 incorporado: %d filas", feedback_rows)

    artefacto_previo = _artefacto_actual(path)
    metrics_previas = artefacto_previo.get("metrics") if artefacto_previo else None
    baseline = metrics_previas if include_feedback and isinstance(metrics_previas, dict) else None
    version_previa = (
        str(artefacto_previo.get("model_version"))
        if artefacto_previo and artefacto_previo.get("model_version")
        else None
    )
    version = siguiente_version(version_previa, con_feedback=include_feedback)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state, stratify=y
    )
    logger.info("Split: train=%d test=%d", len(x_train), len(x_test))

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_proba = pipeline.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test.values, y_pred, y_proba)

    # Gate C4-1: sin degradación → promueve; con degradación → conserva el actual.
    promovido = True
    motivos: list[str] = []
    if include_feedback:
        if baseline:
            promovido, motivos = evaluar_contra_baseline(metrics, baseline, epsilon)
            if promovido:
                logger.info("Gate OK: no degrada la línea base (%s)", json.dumps(baseline))
            else:
                logger.warning("Reentrenamiento NO promovido: %s", "; ".join(motivos))
        else:
            logger.info("Sin artefacto previo: primer entrenamiento con feedback.")

    trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    artifact = {
        "pipeline": pipeline,
        "features": FEATURES,
        "feature_types": FEATURE_TYPES,
        "metrics": metrics,
        # Versión del modelo: semver independiente de `app_version` (C4-1).
        "model_version": version,
        "parent_version": version_previa,
        "trained_at": trained_at,
        "dataset_shape": {"train": int(len(x_train)), "test": int(len(y_test))},
        # Columnas del PredictRequest aún sin datos reales (sin señal hasta producción).
        "missing_in_training": ["Especialidad", "AusenciasPrevias", "CanalRecordatorio"],
        # Valores neutros de entrenamiento para inferencia parcial.
        "defaults": _defaults(x_train),
        # Origen de los datos de este entrenamiento (C4-1).
        "training_data": {"kaggle_rows": kaggle_rows, "feedback_rows": feedback_rows},
        "baseline_metrics": baseline,
    }

    if promovido:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, path)
        _archivar_version(path, artifact)
        logger.info("Artefacto v%s guardado en %s", version, path)
        logger.info("Métricas test: %s", json.dumps(metrics))
    else:
        logger.warning("Artefacto vigente intacto en %s (gate rechazó el nuevo)", path)

    # Comparativo baseline vs. reentrenado (auditoría C4-1), promovido o no.
    _registrar_comparativo(
        path,
        {
            "model_version": version if promovido else version_previa,
            "candidato_version": version,
            "parent_version": version_previa,
            "trained_at": trained_at,
            "promovido": promovido,
            "motivos": motivos,
            "epsilon": epsilon if include_feedback else None,
            "include_feedback": include_feedback,
            "training_data": artifact["training_data"],
            "baseline": baseline,
            "retrained": metrics,
            "delta": _delta_metrics(metrics, baseline),
        },
    )

    return {
        "model_path": str(path),
        "metrics": metrics,
        "promovido": promovido,
        "motivos": motivos,
        "baseline": baseline,
        "feedback_rows": feedback_rows,
        "model_version": version if promovido else version_previa,
        "version_previa": version_previa,
    }


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Entrena el modelo de no-show (C1-1/C4-1)")
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="Incluir feedback real HU-IA-02 con gate de no-degradación (C4-1)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=GATE_EPSILON,
        help=f"Tolerancia del gate (default {GATE_EPSILON})",
    )
    args = parser.parse_args()
    resultado = train(include_feedback=args.feedback, epsilon=args.epsilon)
    if not resultado["promovido"]:
        logger.error("Rechazado por el gate: %s", "; ".join(resultado["motivos"]))
        sys.exit(1)
