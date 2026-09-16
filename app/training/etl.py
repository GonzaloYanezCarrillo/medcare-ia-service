"""ETL del dataset Medical Appointment No Shows (C1-1).

Limpieza reproducible de los datos sucios detectados en C0-3:
- Age < 0 (1 fila con -1) → descartar.
- WaitingDays < 0 (5 filas, AppointmentDay anterior a ScheduledDay) → descartar.
- Handcap>1 → binarizar (una discapacidad o más).
- SMS_received es endógeno (recordatorio dirigido a grupo de riesgo) → se retiene cruda
  en el DataFrame, pero no se usa como feature por defecto.
"""

from __future__ import annotations

from pathlib import Path

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
    """Carga y limpia el dataset, devolviendo (X, y)."""
    df = clean(load_raw(path))
    y = df["No-show"]
    x = df.drop(columns=["PatientId", "AppointmentID", "ScheduledDay", "AppointmentDay", "No-show"])
    return x, y


# Features numéricas de entrada al modelo (sin SMS_received: endógeno).
FEATURES = [
    "Gender",
    "Age",
    "Neighbourhood",
    "Scholarship",
    "Hipertension",
    "Diabetes",
    "Alcoholism",
    "Handcap",
    "WaitingDays",
]
