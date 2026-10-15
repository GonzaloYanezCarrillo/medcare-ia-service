"""Generador del artefacto de demo para el frontend (Kaggle + modelo real).

Toma una muestra estratificada del dataset de no-show, la puntúa con el
artefacto real `models/model.joblib` (mismas features y umbrales de banda que
`PredictService`) y exporta un JSON autocontenido donde cada cita lleva la
etiqueta real (`inasistencia_real`) y el riesgo predicho (`score_riesgo`,
`banda_riesgo`), más un resumen agregado.

No llama a OMRO, a .NET ni a Supabase: es un artefacto offline para que el
frontend lo renderice en una vista de demostración (marcado `"simulacion": true`).

Uso (desde la raíz del proyecto):
    python scripts/generar_demo_citas.py
    python scripts/generar_demo_citas.py --n 60 --seed 42 --out docs/demo/demo_citas.json

Salida:
    docs/demo/demo_citas.json
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# El pipeline imputa NaN→defaults: sklearn avisa al transformar columnas sin
# observaciones en entrenamiento. Ruido esperado en este diagnóstico → silenciar.
warnings.filterwarnings(
    "ignore",
    message=r"Skipping features without any observed values.*",
    category=UserWarning,
)

from sklearn.metrics import roc_auc_score  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.predict_service import _score_to_banda, _score_to_clase  # noqa: E402
from app.training.etl import load_cleaned  # noqa: E402

DEFAULT_OUT = PROJECT_ROOT / "docs" / "demo" / "demo_citas.json"
WEEKDAY_LABELS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def load_artifact() -> dict:
    """Carga y valida el artefacto real (rechaza el dummy de CI)."""
    path = Path(get_settings().model_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or "pipeline" not in artifact:
        raise SystemExit(f"Artefacto inválido en {path}: falta 'pipeline'.")
    if float(artifact.get("metrics", {}).get("auc_roc", 0.0)) <= 0.0:
        raise SystemExit(
            f"El artefacto en {path} parece el dummy de CI (AUC=0). "
            "Genera el modelo real con `python -m app.training.train`."
        )
    return artifact


def sample_stratified(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Muestra aleatoria que preserva la proporción real de no-show."""
    rng = np.random.default_rng(seed)
    pos = df.index[df["no_show"] == 1]
    neg = df.index[df["no_show"] == 0]

    rate = float(df["no_show"].mean())
    n_pos = min(max(1, round(n * rate)), len(pos))
    n_neg = min(n - n_pos, len(neg))

    idx = list(rng.choice(pos, size=n_pos, replace=False)) + list(
        rng.choice(neg, size=n_neg, replace=False)
    )
    return df.loc[idx].sample(frac=1, random_state=seed).reset_index(drop=True)


def build_feature_frame(artifact: dict, sample: pd.DataFrame) -> pd.DataFrame:
    """DataFrame de inferencia: rellena las features sin dato (Kaggle) con defaults."""
    frame = sample[artifact["features"]].copy()
    for col in artifact["features"]:
        if frame[col].isna().all():
            frame[col] = artifact["defaults"][col]

    feature_types = artifact.get("feature_types", {})
    for col in artifact["features"]:
        if feature_types.get(col) == "categorical":
            frame[col] = frame[col].astype(str)
        elif feature_types.get(col) == "numeric":
            frame[col] = frame[col].astype("float64")
    return frame


def build_payload(artifact: dict, sample: pd.DataFrame, scores: np.ndarray) -> dict:
    citas = []
    for i, (_, row) in enumerate(sample.iterrows(), start=1):
        score = float(scores[i - 1])
        weekday = int(row["Weekday"])
        citas.append(
            {
                "cita_id": f"demo-{i:04d}",
                "edad": int(row["Age"]),
                "genero": str(row["Gender"]),
                "dias_espera": int(row["WaitingDays"]),
                "weekday": weekday,
                "dia_semana": WEEKDAY_LABELS[weekday],
                "inasistencia_real": int(row["no_show"]),
                "score_riesgo": round(score, 4),
                "banda_riesgo": _score_to_banda(score),
                "clase": _score_to_clase(score),
            }
        )

    real = [c["inasistencia_real"] for c in citas]
    bandas = Counter(c["banda_riesgo"] for c in citas)
    clases = Counter(c["clase"] for c in citas)
    auc_muestra = None
    if 0 < sum(real) < len(real):
        auc_muestra = round(float(roc_auc_score(real, scores)), 4)

    metrics = artifact.get("metrics", {})
    return {
        "generado_en": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "simulacion": True,
        "fuente": {
            "dataset": Path(get_settings().dataset_path).name,
            "n_total": int(len(sample)),
            "nota": "Muestra del dataset Kaggle; features Especialidad/Ausencias/Canal "
            "no existen en el dataset y usan los defaults del modelo.",
        },
        "modelo": {
            "version": artifact.get("model_version"),
            "auc_test": metrics.get("auc_roc"),
            "sensibilidad": metrics.get("sensibilidad"),
            "especificidad": metrics.get("especificidad"),
        },
        "umbrales_banda": {"Bajo": "< 0.25", "Medio": "0.25 - 0.50", "Alto": "> 0.50"},
        "resumen": {
            "n_citas": len(citas),
            "tasa_inasistencia_real": round(float(np.mean(real)), 4),
            "tasa_inasistencia_predicha": round(float(np.mean(scores)), 4),
            "auc_muestra": auc_muestra,
            "por_banda": {b: int(bandas.get(b, 0)) for b in ("Bajo", "Medio", "Alto")},
            "por_clase": {c: int(clases.get(c, 0)) for c in ("asiste", "no_asiste")},
        },
        "citas": citas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera demo_citas.json para el frontend.")
    parser.add_argument("--n", type=int, default=60, help="N° de citas de la muestra (default 60).")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Ruta de salida del JSON.")
    args = parser.parse_args()

    artifact = load_artifact()
    x, y = load_cleaned()

    df = x.reset_index(drop=True).copy()
    df["no_show"] = y.reset_index(drop=True).values

    sample = sample_stratified(df, args.n, args.seed)
    frame = build_feature_frame(artifact, sample)
    scores = artifact["pipeline"].predict_proba(frame)[:, 1]

    payload = build_payload(artifact, sample, scores)

    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    r = payload["resumen"]
    print(f"Generado: {out}")
    print(
        f"  citas={r['n_citas']} · inasistencia real={r['tasa_inasistencia_real']:.1%} "
        f"· predicha={r['tasa_inasistencia_predicha']:.1%} · AUC muestra={r['auc_muestra']}"
    )
    print(f"  bandas={r['por_banda']} · clases={r['por_clase']}")


if __name__ == "__main__":
    main()
