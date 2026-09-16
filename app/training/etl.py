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
`CanalRecordatorio`) se devuelven como NaN; el pipeline los imputa con un valor neutro
durante el entrenamiento y, además, los `defaults` del artefacto mantienen ese valor. Cuando
haya datos reales de producción (C4-1+), se reentrena con esas columnas ya pobladas y el
imputer deja de intervenir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from app.config import Settings, get_settings

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

    X contiene exactamente las 6 features del contrato; las 3 sin equivalente en el
    dataset se rellenan con NaN (se imputan en el pipeline y persistidas por `defaults`).
    """
    df = clean(load_raw(path))
    y = df["No-show"]

    x = pd.DataFrame(index=df.index)
    x["Age"] = df["Age"]
    x["Gender"] = df["Gender"]
    x["WaitingDays"] = df["WaitingDays"].astype(float)
    # Campos del contrato sin equivalente en el dataset → nulos hasta producción.
    x["Especialidad"] = np.nan
    x["AusenciasPrevias"] = np.nan
    x["CanalRecordatorio"] = np.nan
    return x, y


# Features del modelo = campos del PredictRequest (contrato ia-api.yaml v1.2.0).
FEATURES = [
    "Age",
    "Gender",
    "WaitingDays",
    "Especialidad",
    "AusenciasPrevias",
    "CanalRecordatorio",
]

# Tipo esperado por el pipeline/imputer por feature.
FEATURE_TYPES: dict[str, str] = {
    "Age": "numeric",
    "Gender": "categorical",
    "WaitingDays": "numeric",
    "Especialidad": "categorical",
    "AusenciasPrevias": "numeric",
    "CanalRecordatorio": "categorical",
}
