"""Tests del entrenamiento (C1-1/C1-2).

Cubren `build_pipeline` (ajuste + predicción sobre un frame del contrato), `_defaults`,
`evaluate` y `train` (artefacto joblib + metadatos). `load_cleaned` se parchea con datos
sintéticos para no depender del dataset real de Kaggle en CI.
"""

import joblib
import numpy as np
import pandas as pd
import pytest
from app.training import build_pipeline, train
from app.training.etl import FEATURE_TYPES, FEATURES


def _synthetic_xy(n: int = 400, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Dataset sintético con la misma forma que `load_cleaned` (features del contrato)."""
    rng = np.random.default_rng(seed)
    edad_riesgo = rng.choice([True, False], n, p=[0.45, 0.55])  # correlato artificial
    prob_no_show = 0.12 + 0.60 * edad_riesgo.astype(float)
    y = rng.random(n) < prob_no_show

    x = pd.DataFrame(
        {
            "Age": rng.integers(0, 90, n).astype(float),
            "Gender": rng.choice(["M", "F"], n),
            "WaitingDays": rng.integers(0, 60, n).astype(float),
            "Weekday": rng.integers(0, 7, n).astype(float),
            "Especialidad": np.nan,  # sin datos → sin señal
            "AusenciasPrevias": np.nan,
            "CanalRecordatorio": np.nan,
        }
    )[FEATURES]
    return x, pd.Series(y.astype(int))


def _patch_load_cleaned(monkeypatch, xy):
    import importlib

    train_mod = importlib.import_module("app.training.train")
    monkeypatch.setattr(train_mod, "load_cleaned", lambda path=None: xy)


def test_build_pipeline_ajusta_y_predice_frame_del_contrato(monkeypatch):
    xy = _synthetic_xy()
    _patch_load_cleaned(monkeypatch, xy)

    pipeline = build_pipeline(
        model_kwargs={"n_estimators": 10, "max_depth": 4, "random_state": 0, "n_jobs": 1}
    )
    x, y = xy
    pipeline.fit(x, y)

    proba = pipeline.predict_proba(x.head(5))[:, 1]
    assert proba.shape == (5,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_load_cleaned_devuelve_features_alineadas(monkeypatch):
    x, y = _synthetic_xy()

    assert list(x.columns) == FEATURES
    assert x["Especialidad"].isna().all()
    # balance artificial ~20% positivos (equivalente al 1:4 del dataset real).
    assert 0.15 <= y.mean() <= 0.40


def test_defaults_usan_mediana_moda_y_fill_para_na():
    x, _ = _synthetic_xy()
    from app.training.train import _defaults

    defaults = _defaults(x)
    assert defaults["Age"] == float(x["Age"].median())
    assert defaults["Gender"] == str(x["Gender"].mode()[0])
    # Columnas todo-NaN → el fill neutro del pipeline (no fallan).
    assert defaults["Especialidad"] == "desconocido"
    assert defaults["AusenciasPrevias"] == 0.0
    assert defaults["CanalRecordatorio"] == "desconocido"


def test_evaluate_calcula_metricas():
    from app.training.train import evaluate

    y_true = np.array([1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.2, 0.4, 0.1])

    m = evaluate(y_true, y_pred, y_proba)
    assert 0 <= m["auc_roc"] <= 1
    assert m["sensibilidad"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["especificidad"] == pytest.approx(2 / 3, abs=1e-3)
    assert m["n_test"] == 6


def test_train_genera_artefacto_completo(tmp_path, monkeypatch):
    _patch_load_cleaned(monkeypatch, _synthetic_xy())

    result = train(model_path=str(tmp_path / "model.joblib"), random_state=42)

    assert (tmp_path / "model.joblib").exists()
    assert "model_path" in result and "metrics" in result

    artifact = joblib.load(tmp_path / "model.joblib")
    assert isinstance(artifact, dict)
    assert "pipeline" in artifact
    assert artifact["features"] == FEATURES
    assert artifact["feature_types"] == FEATURE_TYPES
    assert artifact["missing_in_training"] == [
        "Especialidad",
        "AusenciasPrevias",
        "CanalRecordatorio",
    ]
    for key in ("auc_roc", "sensibilidad", "especificidad", "n_test"):
        assert key in artifact["metrics"]
    assert "model_version" in artifact
    assert "dataset_shape" in artifact
    assert "defaults" in artifact
    # El pipeline del artefacto predice sobre un frame del contrato.
    frame = pd.DataFrame([dict(artifact["defaults"])])
    proba = artifact["pipeline"].predict_proba(frame)[0][1]
    assert 0 <= proba <= 1
