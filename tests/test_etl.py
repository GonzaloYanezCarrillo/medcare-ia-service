"""Tests del ETL de entrenamiento (C1-1/C1-2).

Cubren `clean`, `load_raw` (rutas relativas) y `load_cleaned` (features alineadas al
contrato PredictRequest + Weekday, y=binaria) usando datos sintéticos — sin depender
del dataset real de Kaggle (no versionado en CI).
"""

import pandas as pd
import pytest
from app.training.etl import FEATURE_TYPES, FEATURES, clean, load_cleaned, load_raw


def _raw_frame() -> pd.DataFrame:
    """DataFrame crudo pequeño con la estructura del dataset Kaggle."""
    # Aliases cortos: cita el martes 2016-05-03; registros 4, 2 y 6 días antes.
    APPT = "2016-05-03T00:00:00Z"
    cols = [
        "PatientId",
        "AppointmentID",
        "Gender",
        "ScheduledDay",
        "AppointmentDay",
        "Age",
        "Neighbourhood",
        "Scholarship",
        "Hipertension",
        "Diabetes",
        "Alcoholism",
        "Handcap",
        "SMS_received",
        "No-show",
    ]
    rows = [
        # 3 citas válidas: 2 no-show, 1 asistida (WaitingDays 4, 2, 6).
        dict(
            zip(
                cols,
                ["P1", 101, "F", "2016-04-29T10:00:00Z", APPT, 45, "J", 0, 1, 0, 0, 0, 0, "Yes"],
                strict=True,
            )
        ),
        dict(
            zip(
                cols,
                ["P2", 102, "M", "2016-05-01T09:00:00Z", APPT, 30, "C", 1, 0, 0, 0, 2, 1, "No"],
                strict=True,
            )
        ),
        dict(
            zip(
                cols,
                ["P3", 103, "F", "2016-04-27T14:00:00Z", APPT, 22, "V", 0, 0, 0, 1, 0, 1, "Yes"],
                strict=True,
            )
        ),
        # Fila sucia: edad negativa → se descarta.
        dict(
            zip(
                cols,
                ["P4", 104, "M", "2016-05-01T11:00:00Z", APPT, -1, "C", 0, 0, 0, 0, 0, 0, "No"],
                strict=True,
            )
        ),
        # Fila sucia: cita antes del registro (WaitingDays < 0) → se descarta.
        dict(
            zip(
                cols,
                ["P5", 105, "M", "2016-05-05T10:00:00Z", APPT, 40, "P", 0, 0, 0, 0, 0, 0, "No"],
                strict=True,
            )
        ),
    ]
    return pd.DataFrame(rows, columns=cols)


def test_clean_descarta_filas_sucias_y_deriva_features():
    df = clean(_raw_frame())

    # 5 crudas → 3 tras descartar Age<0 y WaitingDays<0.
    assert len(df) == 3
    # WaitingDays derivado: 4, 2, 6 días.
    assert list(df["WaitingDays"]) == [4, 2, 6]
    # Weekday derivado (2016-05-03 es martes → weekday 1).
    assert list(df["Weekday"]) == [1, 1, 1]
    # Objetivo binarizado: Yes→1, No→0.
    assert list(df["No-show"]) == [1, 0, 1]
    # Handcap binarizado (>0 → 1).
    assert list(df["Handcap"]) == [0, 1, 0]


def test_clean_resuelve_rutas_relativas(tmp_path, monkeypatch):
    # `load_raw` debe resolver la ruta relativa contra la raíz del proyecto.
    import app.training.etl as etl

    fake = tmp_path / "fake.csv"
    _raw_frame().to_csv(fake, index=False)
    monkeypatch.setattr(etl, "PROJECT_ROOT", tmp_path)

    df = load_raw(fake.name)
    assert len(df) == 5
    # Ruta absoluta también funciona independientemente de PROJECT_ROOT.
    df2 = load_raw(str(fake))
    assert len(df2) == 5


def test_load_cleaned_alinea_features_al_contrato(tmp_path):
    raw = tmp_path / "raw.csv"
    _raw_frame().to_csv(raw, index=False)

    x, y = load_cleaned(str(raw))

    assert list(x.columns) == FEATURES
    assert len(x) == 3
    assert list(y) == [1, 0, 1]
    # Features del modelo sin datos en el dataset → NaN.
    assert x["Especialidad"].isna().all()
    assert x["AusenciasPrevias"].isna().all()
    assert x["CanalRecordatorio"].isna().all()
    # Tipos declarados coherentes con las columnas.
    assert set(FEATURE_TYPES) == set(FEATURES)
    assert FEATURE_TYPES["Age"] == "numeric"
    assert FEATURE_TYPES["Gender"] == "categorical"


@pytest.mark.parametrize("col", ["Age", "WaitingDays", "Weekday"])
def test_load_cleaned_columna_desconocida_fuera_de_features(col):
    assert col in FEATURES
