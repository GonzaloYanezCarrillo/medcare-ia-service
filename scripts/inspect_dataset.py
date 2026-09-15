"""Inspección reproducible del dataset Medical Appointment No Shows (C0-3).

Validación de: forma, nulos, variable objetivo, desbalanceo y datos sucios
que deben limpiarse en el ETL (C1-1). Uso: python scripts/inspect_dataset.py
"""

import pandas as pd
from app.config import get_settings

settings = get_settings()
DATA_PATH = settings.dataset_path


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    print(f"=== Dataset: {DATA_PATH} ===")
    print(f"filas={df.shape[0]}  columnas={df.shape[1]}  nulos={int(df.isna().sum().sum())}")
    print()
    print("=== Variable objetivo No-show ===")
    print(df["No-show"].value_counts())
    no_show_ratio = (df["No-show"] == "Yes").mean()
    print(f"proporcion no_show = {no_show_ratio:.3f}")
    print()
    print("=== Datos sucios (a limpiar en C1-1) ===")
    print(
        f"Age negativos: {int((df['Age'] < 0).sum())}  (rango {df['Age'].min()}..{df['Age'].max()})"
    )
    print(f"Handcap > 1: {int((df['Handcap'] > 1).sum())}")
    sch = pd.to_datetime(df["ScheduledDay"], format="mixed", utc=True)
    app = pd.to_datetime(df["AppointmentDay"], format="mixed", utc=True)
    wait = (app.dt.normalize() - sch.dt.normalize()).dt.days
    print(f"WaitingDays negativos: {int((wait < 0).sum())}")


if __name__ == "__main__":
    main()
