"""Exploración gráfica del modelo de no-show (diagnóstico offline, sin tocar el servicio).

Carga el artefacto real `models/model.joblib`, construye consultas sintéticas
basadas en el contrato `PredictRequest`, y genera gráficos PNG en `docs/graficos/`
más un reporte Markdown (`docs/graficos/REPORTE.md`).

Uso (desde la raíz del proyecto):
    python scripts/explore_model.py

Salidas:
    docs/graficos/01_curva_dias_espera.png
    docs/graficos/02_curva_edad.png
    docs/graficos/03_curva_ausencias_previas.png
    docs/graficos/04_curva_weekday.png
    docs/graficos/05_heatmap_waiting_ausencias.png
    docs/graficos/06_bandas_frecuencia.png
    docs/graficos/07_importancia_features.png
    docs/graficos/REPORTE.md

Avisos: si el artefacto es el dummy de CI (AUC=0) el script se detiene: no sirve
para exploración real. Las features sin datos de entrenamiento (Especialidad,
AusenciasPrevias, CanalRecordatorio) usan los `defaults` del artefacto.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # backend sin GUI: guarda a archivo, no abre ventanas.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# El pipeline imputa NaN→defaults: sklearn avisa (esperado) al transformar columnas
# que en entrenamiento no tenían observaciones. Ruido en la salida diagnóstica → silenciar.
warnings.filterwarnings(
    "ignore",
    message=r"Skipping features without any observed values.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402

OUT_DIR = PROJECT_ROOT / "docs" / "graficos"

WEEKDAY_LABELS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

# Consulta base "típica" según el contrato (valores sensatos de un paciente real).
BASE_REQUEST = {
    "edad": 42,
    "genero": "F",
    "dias_espera": 15,
    "especialidad": "medicina_general",
    "ausencias_previas": 2,
    "canal_recordatorio": "ninguno",
}

_REQUEST_TO_FEATURE = {
    "edad": "Age",
    "genero": "Gender",
    "dias_espera": "WaitingDays",
    "especialidad": "Especialidad",
    "ausencias_previas": "AusenciasPrevias",
    "canal_recordatorio": "CanalRecordatorio",
}


def _artifact() -> dict:
    path = PROJECT_ROOT / get_settings().model_path
    if not path.exists():
        raise SystemExit(
            f"No existe el artefacto del modelo en {path}. Ejecuta `python -m app.training.train`."
        )
    artifact = joblib.load(path)
    auc = (artifact.get("metrics") or {}).get("auc_roc", 0)
    if isinstance(auc, (int, float)) and auc == 0:
        raise SystemExit(
            "El artefacto en {path} es el dummy de CI (AUC=0). Genera el modelo real "
            "con `python -m app.training.train` antes de explorar."
        )
    return artifact


def build_frame(artifact: dict, rows: list[dict]) -> pd.DataFrame:
    """Construye un DataFrame de inferencia a partir de defaults + overrides.

    Misma lógica de coacción de tipos que `PredictService._build_feature_frame`.
    `rows` es una lista de dicts: cada uno es un subset de las features del contrato.
    """
    base = dict(artifact["defaults"])
    frames = []
    for row in rows:
        r = dict(base)
        for req_field, feature in _REQUEST_TO_FEATURE.items():
            if req_field in row:
                r[feature] = row[req_field]
        frames.append(pd.DataFrame([r], columns=artifact["features"]))
    frame = pd.concat(frames, ignore_index=True)
    feature_types = artifact.get("feature_types", {})
    for col in artifact["features"]:
        if feature_types.get(col) == "categorical":
            frame[col] = frame[col].astype(str)
        elif feature_types.get(col) == "numeric":
            frame[col] = frame[col].astype("float64")
    return frame


def proba(artifact: dict, row: dict) -> float:
    frame = build_frame(artifact, [row])
    return float(artifact["pipeline"].predict_proba(frame)[0][1])


def _save(fig, name: str) -> Path:
    out = OUT_DIR / name
    fig.savefig(out, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out


def plot_heatmap(artifact: dict) -> Path:
    """Heatmap probabilidad variando dos features (días de espera × ausencias)."""
    dias = list(range(0, 61, 5))
    ausencias = list(range(0, 11))
    rows = []
    for d in dias:
        for a in ausencias:
            row = dict(BASE_REQUEST)
            row["dias_espera"] = d
            row["ausencias_previas"] = a
            rows.append(row)
    frame = build_frame(artifact, rows)
    probas = artifact["pipeline"].predict_proba(frame)[:, 1].reshape(len(ausencias), len(dias))

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(probas, aspect="auto", origin="lower", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(dias)), dias)
    ax.set_yticks(range(len(ausencias)), ausencias)
    ax.set_xlabel("Días de espera")
    ax.set_ylabel("Ausencias previas")
    ax.set_title("Probabilidad de no-show (días de espera × ausencias previas)")
    fig.colorbar(im, ax=ax, label="Probabilidad")
    return _save(fig, "05_heatmap_waiting_ausencias.png")


def plot_bandas_frecuencia(artifact: dict) -> Path:
    """Frecuencia de bandas sobre un barrido de consultas realistas."""
    scenarios = []
    for d in range(0, 61, 5):
        for a in range(0, 11, 2):
            row = dict(BASE_REQUEST)
            row["dias_espera"] = d
            row["ausencias_previas"] = a
            scenarios.append(row)
    frame = build_frame(artifact, scenarios)
    probas = artifact["pipeline"].predict_proba(frame)[:, 1]
    bandas = np.where(probas < 0.25, "Bajo", np.where(probas <= 0.5, "Medio", "Alto"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    counts = {b: int((bandas == b).sum()) for b in ("Bajo", "Medio", "Alto")}
    ax1.bar(counts.keys(), counts.values(), color=["#55A868", "#C44E52", "#8172B2"])
    ax1.set_title("Frecuencia de bandas (barrido)")
    ax1.set_ylabel("N° de escenarios")

    clases = np.where(probas >= 0.5, "no_asiste", "asiste")
    cl_counts = {c: int((clases == c).sum()) for c in ("asiste", "no_asiste")}
    ax2.bar(cl_counts.keys(), cl_counts.values(), color=["#4C72B0", "#DD8452"])
    ax2.set_title("Clase discreta (umbral 0.5)")
    ax2.set_ylabel("N° de escenarios")
    return _save(fig, "06_bandas_frecuencia.png")


def plot_importancia(artifact: dict) -> Path:
    """Importancia de features del RandomForest (sobre columnas transformadas)."""
    pipeline = artifact["pipeline"]
    preprocessor = pipeline.named_steps["prep"]
    clf = pipeline.named_steps["clf"]
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = list(pipeline.feature_names_in_)
    importances = np.asarray(clf.feature_importances_)

    order = np.argsort(importances)[-15:]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([names[i] for i in order], importances[order], color="#4C72B0")
    ax.set_title("Importancia de features (RandomForest)")
    ax.set_xlabel("Importancia")
    return _save(fig, "07_importancia_features.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = _artifact()
    logs: list[tuple[str, Path]] = []

    # 1. Curva días de espera.
    dias = list(range(0, 61, 3))
    rows = []
    for d in dias:
        row = dict(BASE_REQUEST)
        row["dias_espera"] = d
        rows.append(row)
    frame = build_frame(artifact, rows)
    probas = artifact["pipeline"].predict_proba(frame)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dias, probas, marker="o", color="#4C72B0")
    ax.axhline(0.25, color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Probabilidad de no-show vs días de espera")
    ax.set_xlabel("Días de espera")
    ax.set_ylabel("Probabilidad de no-show")
    logs.append(("01_curva_dias_espera.png", _save(fig, "01_curva_dias_espera.png")))

    # 2. Curva edad.
    edades = list(range(10, 91, 5))
    rows = []
    for e in edades:
        row = dict(BASE_REQUEST)
        row["edad"] = e
        rows.append(row)
    frame = build_frame(artifact, rows)
    probas = artifact["pipeline"].predict_proba(frame)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(edades, probas, marker="o", color="#55A868")
    ax.axhline(0.25, color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Probabilidad de no-show vs edad")
    ax.set_xlabel("Edad")
    ax.set_ylabel("Probabilidad de no-show")
    logs.append(("02_curva_edad.png", _save(fig, "02_curva_edad.png")))

    # 3. Curva ausencias previas.
    aus = list(range(0, 11))
    rows = []
    for a in aus:
        row = dict(BASE_REQUEST)
        row["ausencias_previas"] = a
        rows.append(row)
    frame = build_frame(artifact, rows)
    probas = artifact["pipeline"].predict_proba(frame)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(aus, probas, marker="o", color="#DD8452")
    ax.axhline(0.25, color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Probabilidad de no-show vs ausencias previas")
    ax.set_xlabel("Ausencias previas")
    ax.set_ylabel("Probabilidad de no-show")
    logs.append(("03_curva_ausencias_previas.png", _save(fig, "03_curva_ausencias_previas.png")))

    # 4. Día de la semana (7 puntos).
    rows = []
    for w in range(7):
        r = dict(artifact["defaults"])
        r["Weekday"] = float(w)
        for req_field, feature in _REQUEST_TO_FEATURE.items():
            if req_field in BASE_REQUEST:
                r[feature] = BASE_REQUEST[req_field]
        rows.append(r)
    frame = pd.DataFrame(rows, columns=artifact["features"])
    feature_types = artifact.get("feature_types", {})
    for col in artifact["features"]:
        if feature_types.get(col) == "categorical":
            frame[col] = frame[col].astype(str)
        elif feature_types.get(col) == "numeric":
            frame[col] = frame[col].astype("float64")
    probas = artifact["pipeline"].predict_proba(frame)[:, 1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(7), probas, color="#8172B2")
    ax.set_xticks(range(7), WEEKDAY_LABELS)
    ax.axhline(0.25, color="orange", linestyle=":", linewidth=0.8)
    ax.axhline(0.5, color="red", linestyle=":", linewidth=0.8)
    ax.set_ylim(0, 1)
    ax.set_title("Probabilidad de no-show por día de la semana")
    ax.set_xlabel("Día de la semana")
    ax.set_ylabel("Probabilidad")
    logs.append(("04_curva_weekday.png", _save(fig, "04_curva_weekday.png")))

    # 5. Heatmap.
    logs.append(("05_heatmap_waiting_ausencias.png", plot_heatmap(artifact)))

    # 6. Bandas y clase.
    logs.append(("06_bandas_frecuencia.png", plot_bandas_frecuencia(artifact)))

    # 7. Importancia.
    logs.append(("07_importancia_features.png", plot_importancia(artifact)))

    # Reporte Markdown.
    lines = [
        "# Reporte de exploración del modelo de no-show",
        "",
        f"Artefacto: `{get_settings().model_path}` · versión {artifact.get('model_version')}",
        "Métricas (test): "
        f"AUC {artifact['metrics']['auc_roc']} · Sensibilidad "
        f"{artifact['metrics']['sensibilidad']} · Especificidad "
        f"{artifact['metrics']['especificidad']}",
        "",
        "Consulta base del reporte: `"
        + ", ".join(f"{k}={v}" for k, v in BASE_REQUEST.items())
        + "`",
        "",
        "## Ejemplos de consultas",
        "",
        "| Escenario | dias_espera | ausencias_previas | Probabilidad | Banda | Clase |",
        "|-----------|-------------|-------------------|--------------|-------|-------|",
    ]
    demo = [
        ("Paciente joven, cita rápida", 2, 0),
        ("Cita promedio", 15, 2),
        ("Espera larga", 40, 2),
        ("Muchas ausencias", 20, 7),
        ("Peor caso", 60, 10),
    ]
    for desc, d, a in demo:
        row = dict(BASE_REQUEST)
        row["dias_espera"] = d
        row["ausencias_previas"] = a
        p = proba(artifact, row)
        banda = "Bajo" if p < 0.25 else ("Medio" if p <= 0.5 else "Alto")
        clase = "no_asiste" if p >= 0.5 else "asiste"
        lines.append(f"| {desc} | {d} | {a} | {p:.3f} | {banda} | {clase} |")
    lines.append("")
    for name, _ in logs:
        lines.append(f"## {name}")
        lines.append(f"![{name}](graficos/{name})")
        lines.append("")
    report = OUT_DIR / "REPORTE.md"
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Gráficos generados en {OUT_DIR}:")
    for name, _ in logs:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
