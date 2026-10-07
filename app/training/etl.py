"""ETL del dataset Medical Appointment No Shows (C1-1).

Limpieza reproducible de los datos sucios detectados en C0-3:
- Age < 0 (1 fila con -1) → descartar.
- WaitingDays < 0 (5 filas, AppointmentDay anterior a ScheduledDay) → descartar.
- Handcap>1 → binarizar (una discapacidad o más).
- SMS_received es endógeno (recordatorio dirigido a grupo de riesgo) → se retiene cruda
  en el DataFrame, pero no se usa como feature por defecto.

Alineación con el contrato `PredictRequest` (C1-1): `load_cleaned` devuelve **solo los 6
campos del request** (edad, género, días de espera, especialidad, ausencias previas y canal
de recordatorio). Los 3 que el dataset no expone (`Especialidad`, `AusenciasPrevias`,
`CanalRecordatorio`) se devuelven como NaN: **no aportan señal al modelo** mientras no haya
datos reales (sklearn las maneja como missing en entrenamiento; en inferencia se rellenan
desde `defaults`). Cuando el modelo se reentrene con datos reales de producción (C4-1+),
esas columnas vendrán pobladas y empezarán a aportar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings, get_settings
from app.services.feedback_store import FeedbackStore

# Raíz del proyecto (app/training/etl.py → 3 niveles arriba).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_raw(path: str | None = None) -> pd.DataFrame:
    """Carga el CSV crudo del dataset.

    Las rutas relativas se resuelven contra la raíz del proyecto (no el CWD),
    para que el entrenamiento funcione desde cualquier directorio.
    """
    settings: Settings = get_settings()
    p = Path(path or settings.dataset_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return pd.read_csv(p)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica la limpieza de datos sucios documentada en C0-3."""
    df = df.copy()

    # 1. Fechas → derivar WaitingDays (días entre registro y cita).
    scheduled = pd.to_datetime(df["ScheduledDay"], format="mixed", utc=True)
    appointment = pd.to_datetime(df["AppointmentDay"], format="mixed", utc=True)
    df["WaitingDays"] = (appointment.dt.normalize() - scheduled.dt.normalize()).dt.days

    # 1b. Día de la semana de la cita (0=Lunes ... 6=Domingo), feature C1-2.
    df["Weekday"] = appointment.dt.weekday

    # 2. Edad negativa → descartar fila.
    df = df[df["Age"] >= 0]

    # 3. Cita programada antes del registro → descartar (WaitingDays < 0).
    df = df[df["WaitingDays"] >= 0]

    # 4. Variable objetivo binaria 0/1 (No-show).
    df["No-show"] = (df["No-show"] == "Yes").astype(int)

    # 5. Handcap → binario (una discapacidad o más).
    df["Handcap"] = (df["Handcap"] > 0).astype(int)

    return df


def load_cleaned(path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Carga y limpia el dataset, devolviendo (X, y) alineado al PredictRequest.

    X contiene las features del contrato; las 3 sin equivalente en el dataset se
    rellenan con NaN y quedan sin señal hasta que haya datos reales de producción.
    """
    df = clean(load_raw(path))
    y = df["No-show"]

    x = pd.DataFrame(index=df.index)
    x["Age"] = df["Age"]
    x["Gender"] = df["Gender"]
    x["WaitingDays"] = df["WaitingDays"].astype(float)
    x["Weekday"] = df["Weekday"].astype(float)
    # Campos del contrato sin equivalente en el dataset → nulos hasta producción.
    x["Especialidad"] = np.nan
    x["AusenciasPrevias"] = np.nan
    x["CanalRecordatorio"] = np.nan
    return x, y


def load_feedback(path: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Carga el feedback de asistencia real (C4-1 / HU-IA-02) alineado al modelo.

    Devuelve (X, y) con las mismas `FEATURES` que `load_cleaned`, para poder
    concatenarlos y reentrenar. Semántica de la etiqueta: en el dataset Kaggle
    `y=1` es no-show, así que se invierte `asistio` (`asistio=False` → y=1).

    RNF-SEG-03: el JSONL solo contiene features pre-cita + etiqueta; X se
    construye únicamente con `FEATURES` (el label y los metadatos nunca entran
    como features → sin leakage desde el feedback).
    """
    settings: Settings = get_settings()
    store = FeedbackStore(path or settings.feedback_path)
    registros = store.registros()
    if not registros:
        return pd.DataFrame(columns=FEATURES), pd.Series(dtype=int)

    df = pd.DataFrame(registros)
    x = pd.DataFrame(index=df.index)
    x["Age"] = pd.to_numeric(df["edad"], errors="coerce")
    x["Gender"] = df["genero"].astype(str)
    x["WaitingDays"] = pd.to_numeric(df["dias_espera"], errors="coerce")
    # Weekday deriva de la fecha de la cita (misma derivación que en el ETL Kaggle).
    x["Weekday"] = pd.to_datetime(df["fecha_cita"], errors="coerce").dt.weekday.astype(float)
    x["Especialidad"] = df["especialidad"].astype(str)
    x["AusenciasPrevias"] = pd.to_numeric(df["ausencias_previas"], errors="coerce")
    canal = df["canal_recordatorio"] if "canal_recordatorio" in df.columns else "ninguno"
    x["CanalRecordatorio"] = canal.fillna("ninguno").astype(str)

    # Inversión de la etiqueta: y=1 → no-show (igual que el dataset Kaggle).
    y = (~df["asistio"].astype(bool)).astype(int)
    return x[FEATURES], y


# Features del modelo = campos del PredictRequest (contrato ia-api.yaml v1.2.0)
# + feature derivada Weekday (C1-2, no expuesta en el contrato → default en inferencia).
FEATURES = [
    "Age",
    "Gender",
    "WaitingDays",
    "Weekday",
    "Especialidad",
    "AusenciasPrevias",
    "CanalRecordatorio",
]

# Tipo esperado por el pipeline/imputer por feature.
FEATURE_TYPES: dict[str, str] = {
    "Age": "numeric",
    "Gender": "categorical",
    "WaitingDays": "numeric",
    "Weekday": "numeric",
    "Especialidad": "categorical",
    "AusenciasPrevias": "numeric",
    "CanalRecordatorio": "categorical",
}
