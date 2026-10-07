"""Tests del reentrenamiento con feedback, gate de no-degradación y versionado (C4-1).

Cubren `load_feedback` (alineación a FEATURES, inversión de etiqueta, vacío),
`evaluar_contra_baseline` (tolerancias), el flujo completo de `train(
include_feedback=True)` (promoción, bloqueo, conservación del artefacto vigente)
y el versionado semver del modelo (`siguiente_version`, linaje, `versions/<v>/`,
`latest.json`) y el comparativo `metrics.json` (baseline vs. reentrenado). El
split/fit se parchea para no depender de Kaggle ni datos reales.
"""

import importlib
import json

import joblib
import pytest
from app.training.etl import FEATURES, load_feedback
from app.training.train import evaluar_contra_baseline
from tests.test_train import _synthetic_xy

train_mod = importlib.import_module("app.training.train")


def _feedback_record(**campos) -> dict:
    registro = {
        "cita_id": "550e8400-e29b-41d4-a716-446655440000",
        "asistio": True,
        "fecha_cita": "2026-10-05",  # lunes → Weekday=0
        "edad": 34,
        "genero": "F",
        "dias_espera": 5,
        "especialidad": "psicologia",
        "ausencias_previas": 1,
        "canal_recordatorio": "whatsapp",
        "score_riesgo": 0.63,
        "registrado_en": "2026-10-05T12:00:00+00:00",
    }
    registro.update(campos)
    return registro


def _write_feedback(tmp_path, registros: list[dict]) -> str:
    from app.services.feedback_store import FeedbackStore

    store = FeedbackStore(str(tmp_path / "asistencia.jsonl"))
    for r in registros:
        store.registrar(r)
    return store.path


# ---------------------------------------------------------------------------
# load_feedback (ETL del feedback)
# ---------------------------------------------------------------------------


def test_load_feedback_alinea_features(tmp_path):
    path = _write_feedback(tmp_path, [_feedback_record(), _feedback_record(cita_id="otra")])
    x, y = load_feedback(path)

    assert list(x.columns) == FEATURES
    assert len(x) == 2
    assert x["Age"].tolist() == [34.0, 34.0]
    assert x["WaitingDays"].tolist() == [5.0, 5.0]
    assert x["AusenciasPrevias"].tolist() == [1.0, 1.0]
    assert x["Especialidad"].tolist() == ["psicologia", "psicologia"]
    assert x["CanalRecordatorio"].tolist() == ["whatsapp", "whatsapp"]
    # 2026-10-05 es lunes → Weekday=0 (misma derivación que el ETL Kaggle).
    assert x["Weekday"].tolist() == [0.0, 0.0]
    assert y.tolist() == [0, 0]


def test_load_feedback_invierte_etiqueta(tmp_path):
    """y=1 = no-show: asistio=False → 1; asistio=True → 0 (semántica Kaggle)."""
    path = _write_feedback(
        tmp_path,
        [
            _feedback_record(cita_id="a", asistio=False),
            _feedback_record(cita_id="b", asistio=True),
        ],
    )
    _, y = load_feedback(path)
    assert y.tolist() == [1, 0]


def test_load_feedback_vacio(tmp_path):
    """Archivo inexistente → frame vacío con las columnas esperadas (concat seguro)."""
    x, y = load_feedback(str(tmp_path / "no_existe.jsonl"))
    assert list(x.columns) == FEATURES
    assert len(x) == 0
    assert len(y) == 0


def test_feedback_x_solo_features_sin_label(tmp_path):
    """Leakage: X del feedback solo contiene FEATURES; etiqueta y metadatos fuera."""
    path = _write_feedback(tmp_path, [_feedback_record()])
    x, y = load_feedback(path)

    for columna_no_feature in ("asistio", "score_riesgo", "cita_id", "registrado_en"):
        assert columna_no_feature not in x.columns
    # El label viaja solo en y, nunca como feature.
    assert len(y) == len(x) == 1


# ---------------------------------------------------------------------------
# evaluar_contra_baseline (gate)
# ---------------------------------------------------------------------------


def test_gate_promueve_sin_degradacion():
    base = {"auc_roc": 0.7174, "sensibilidad": 0.7829, "especificidad": 0.5490}
    nuevas = {"auc_roc": 0.73, "sensibilidad": 0.80, "especificidad": 0.50}
    promovido, motivos = evaluar_contra_baseline(nuevas, base)
    assert promovido is True
    assert motivos == []


def test_gate_tolerancia_epsilon():
    base = {"auc_roc": 0.7174, "sensibilidad": 0.7829}
    # Caída dentro de la tolerancia (0.005 < 0.01) → promueve.
    dentro = {"auc_roc": 0.7124, "sensibilidad": 0.7779}
    assert evaluar_contra_baseline(dentro, base)[0] is True
    # Caída más allá de la tolerancia → bloquea.
    fuera = {"auc_roc": 0.7000, "sensibilidad": 0.7829}
    promovido, motivos = evaluar_contra_baseline(fuera, base)
    assert promovido is False
    assert len(motivos) == 1 and "auc_roc" in motivos[0]


def test_gate_bloquea_por_sensibilidad():
    base = {"auc_roc": 0.80, "sensibilidad": 0.75}
    nuevas = {"auc_roc": 0.85, "sensibilidad": 0.70}
    promovido, motivos = evaluar_contra_baseline(nuevas, base)
    assert promovido is False
    assert "sensibilidad" in motivos[0]


def test_gate_no_bloquea_por_especificidad_sola():
    """El gate es AUC + sensibilidad (prioridad del triaje); especificidad no bloquea."""
    base = {"auc_roc": 0.80, "sensibilidad": 0.75, "especificidad": 0.90}
    nuevas = {"auc_roc": 0.81, "sensibilidad": 0.76, "especificidad": 0.40}
    assert evaluar_contra_baseline(nuevas, base)[0] is True


# ---------------------------------------------------------------------------
# train(include_feedback=True): flujo completo
# ---------------------------------------------------------------------------


def _baseline(path, auc: float, sens: float, version: str = "1.0.0") -> None:
    joblib.dump(
        {
            "metrics": {"auc_roc": auc, "sensibilidad": sens, "n_test": 10},
            "model_version": version,
        },
        path,
    )


def _patch_retrain(monkeypatch, xy, metrics_nuevos: dict):
    x, y = xy
    monkeypatch.setattr(train_mod, "load_cleaned", lambda path=None: xy)
    monkeypatch.setattr(train_mod, "load_feedback", lambda path=None: (x, y))
    monkeypatch.setattr(train_mod, "evaluate", lambda *args, **kwargs: metrics_nuevos)


def test_retrain_promueve_y_sobrescribe(tmp_path, monkeypatch):
    """Sin degradación → escribe el artefacto con las métricas nuevas."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.70, sens=0.70)
    nuevos = {"auc_roc": 0.75, "sensibilidad": 0.80, "especificidad": 0.55, "n_test": 80}
    _patch_retrain(monkeypatch, _synthetic_xy(), nuevos)

    resultado = train_mod.train(model_path=str(modelo), include_feedback=True)

    assert resultado["promovido"] is True
    assert resultado["motivos"] == []
    assert resultado["feedback_rows"] > 0
    artifact = joblib.load(modelo)
    assert artifact["metrics"] == nuevos
    assert artifact["training_data"]["feedback_rows"] > 0
    assert artifact["baseline_metrics"]["auc_roc"] == 0.70


def test_retrain_bloquea_degradacion_no_sobrescribe(tmp_path, monkeypatch):
    """Con degradación → NO escribe; el artefacto vigente queda intacto."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.80, sens=0.80)
    bytes_antes = modelo.read_bytes()
    degradados = {"auc_roc": 0.50, "sensibilidad": 0.85, "especificidad": 0.90, "n_test": 80}
    _patch_retrain(monkeypatch, _synthetic_xy(), degradados)

    resultado = train_mod.train(model_path=str(modelo), include_feedback=True)

    assert resultado["promovido"] is False
    assert resultado["motivos"]  # motivo de degradación
    assert "auc_roc" in resultado["motivos"][0]
    assert modelo.read_bytes() == bytes_antes  # vigente intacto
    # El candidato y la baseline sí quedan visibles para auditoría.
    assert resultado["metrics"] == degradados
    assert resultado["baseline"]["auc_roc"] == 0.80


def test_retrain_sin_baseline_previa_promueve(tmp_path, monkeypatch):
    """Primer entrenamiento con feedback (sin artefacto previo) → promueve."""
    modelo = tmp_path / "nuevo.joblib"
    nuevos = {"auc_roc": 0.72, "sensibilidad": 0.79, "especificidad": 0.55, "n_test": 80}
    _patch_retrain(monkeypatch, _synthetic_xy(), nuevos)

    resultado = train_mod.train(model_path=str(modelo), include_feedback=True)

    assert resultado["promovido"] is True
    assert resultado["baseline"] is None
    assert modelo.exists()


def test_retrain_sin_feedback_no_toca_gate(tmp_path, monkeypatch):
    """Reentrenamiento Kaggle clásico: sin gate, siempre escribe (comportamiento C1-1)."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.99, sens=0.99)
    nuevos = {"auc_roc": 0.50, "sensibilidad": 0.50, "especificidad": 0.50, "n_test": 80}
    _patch_retrain(monkeypatch, _synthetic_xy(), nuevos)

    resultado = train_mod.train(model_path=str(modelo), include_feedback=False)

    assert resultado["promovido"] is True
    assert resultado["feedback_rows"] == 0
    assert resultado["baseline"] is None
    assert joblib.load(modelo)["metrics"] == nuevos


# ---------------------------------------------------------------------------
# Versionado semver del modelo (C4-1, Paso 5)
# ---------------------------------------------------------------------------

_MEJORES = {"auc_roc": 0.75, "sensibilidad": 0.80, "especificidad": 0.55, "n_test": 80}
_PEORES = {"auc_roc": 0.50, "sensibilidad": 0.85, "especificidad": 0.90, "n_test": 80}


def test_siguiente_version_reglas():
    from app.training.train import siguiente_version

    assert siguiente_version(None, con_feedback=False) == "1.0.0"
    assert siguiente_version(None, con_feedback=True) == "1.0.0"
    assert siguiente_version("1.2.3", con_feedback=False) == "1.2.4"  # patch
    assert siguiente_version("1.2.3", con_feedback=True) == "1.3.0"  # menor (datos nuevos)
    assert siguiente_version("no-semver", con_feedback=True) == "1.0.0"


def test_version_independiente_de_app_version(tmp_path, monkeypatch):
    """model_version es semver propio del modelo, no `app_version` de la API."""
    from app.config import get_settings

    modelo = tmp_path / "model.joblib"
    _patch_retrain(monkeypatch, _synthetic_xy(), _MEJORES)

    train_mod.train(model_path=str(modelo))

    artifact = joblib.load(modelo)
    assert artifact["model_version"] == "1.0.0"
    assert artifact["model_version"] != get_settings().app_version
    assert artifact["parent_version"] is None
    assert artifact["trained_at"]


def test_retrain_feedback_bump_menor_y_linaje(tmp_path, monkeypatch):
    """Feedback = datos nuevos → bump menor; linaje en parent_version/trained_at."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.70, sens=0.70, version="1.2.3")
    _patch_retrain(monkeypatch, _synthetic_xy(), _MEJORES)

    resultado = train_mod.train(model_path=str(modelo), include_feedback=True)

    assert resultado["model_version"] == "1.3.0"
    artifact = joblib.load(modelo)
    assert artifact["model_version"] == "1.3.0"
    assert artifact["parent_version"] == "1.2.3"
    assert artifact["trained_at"].startswith("20")


def test_retrain_sin_feedback_bump_patch(tmp_path, monkeypatch):
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.70, sens=0.70, version="1.2.3")
    _patch_retrain(monkeypatch, _synthetic_xy(), _MEJORES)

    resultado = train_mod.train(model_path=str(modelo), include_feedback=False)

    assert resultado["model_version"] == "1.2.4"


def test_archivo_versions_y_latest_json(tmp_path, monkeypatch):
    """Al promover se archiva en versions/<v>/ y latest.json apunta a esa versión."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.70, sens=0.70, version="1.0.0")
    _patch_retrain(monkeypatch, _synthetic_xy(), _MEJORES)

    train_mod.train(model_path=str(modelo), include_feedback=True)

    archivado = tmp_path / "versions" / "1.1.0" / "model.joblib"
    assert archivado.exists()
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert latest["model_version"] == "1.1.0"
    assert latest["parent_version"] == "1.0.0"
    assert latest["path"] == "versions/1.1.0/model.joblib"
    assert latest["metrics"] == _MEJORES
    assert latest["training_data"]["feedback_rows"] > 0
    assert joblib.load(archivado)["model_version"] == "1.1.0"


def test_gate_rechazado_no_versiona(tmp_path, monkeypatch):
    """Si el gate bloquea: sin bump de versión, sin archivo, sin latest.json."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.80, sens=0.80, version="2.1.0")
    _patch_retrain(monkeypatch, _synthetic_xy(), _PEORES)

    resultado = train_mod.train(model_path=str(modelo), include_feedback=True)

    assert resultado["promovido"] is False
    assert resultado["model_version"] == "2.1.0"
    assert not (tmp_path / "versions").exists()
    assert not (tmp_path / "latest.json").exists()
    assert joblib.load(modelo)["model_version"] == "2.1.0"


# ---------------------------------------------------------------------------
# Comparativo baseline vs. reentrenado en metrics.json (C4-1, Paso 6)
# ---------------------------------------------------------------------------


def _metrics_json(tmp_path) -> dict:
    return json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))


def test_metrics_json_comparativo_promocion(tmp_path, monkeypatch):
    """metrics.json registra baseline, retrained, delta y resultado del gate."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.70, sens=0.70, version="1.0.0")
    _patch_retrain(monkeypatch, _synthetic_xy(), _MEJORES)

    train_mod.train(model_path=str(modelo), include_feedback=True)

    reporte = _metrics_json(tmp_path)
    assert reporte["promovido"] is True
    assert reporte["motivos"] == []
    assert reporte["model_version"] == "1.1.0"
    assert reporte["parent_version"] == "1.0.0"
    assert reporte["include_feedback"] is True
    assert reporte["epsilon"] == pytest.approx(0.01)
    assert reporte["baseline"]["auc_roc"] == 0.70
    assert reporte["retrained"] == _MEJORES
    assert reporte["delta"]["auc_roc"] == pytest.approx(0.05)
    assert reporte["delta"]["sensibilidad"] == pytest.approx(0.10)
    assert reporte["training_data"]["feedback_rows"] > 0


def test_metrics_json_registra_rechazo(tmp_path, monkeypatch):
    """Un rechazo del gate también queda documentado en metrics.json."""
    modelo = tmp_path / "model.joblib"
    _baseline(modelo, auc=0.80, sens=0.80, version="2.1.0")
    _patch_retrain(monkeypatch, _synthetic_xy(), _PEORES)

    train_mod.train(model_path=str(modelo), include_feedback=True)

    reporte = _metrics_json(tmp_path)
    assert reporte["promovido"] is False
    assert reporte["model_version"] == "2.1.0"  # versión vigente = previa
    assert reporte["candidato_version"] == "2.2.0"
    assert reporte["delta"]["auc_roc"] == pytest.approx(-0.30)
    assert reporte["motivos"]


def test_metrics_json_sin_feedback_sin_baseline(tmp_path, monkeypatch):
    """Entrenamiento Kaggle puro: sin baseline/delta (no aplica el gate)."""
    modelo = tmp_path / "model.joblib"
    _patch_retrain(monkeypatch, _synthetic_xy(), _MEJORES)

    train_mod.train(model_path=str(modelo), include_feedback=False)

    reporte = _metrics_json(tmp_path)
    assert reporte["include_feedback"] is False
    assert reporte["baseline"] is None
    assert reporte["delta"] is None
    assert reporte["promovido"] is True
