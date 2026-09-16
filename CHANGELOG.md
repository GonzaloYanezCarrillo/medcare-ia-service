# Changelog — MedCare IA (`ia_service`)

Todas las modificaciones de este repositorio se documentan aquí, siguiendo [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y [SemVer](https://semver.org/lang/es/).

El servicio implementa el contrato [`ia-api.yaml`](https://github.com/pabloordenes/medcare-contracts) (owner: Dev C).

## [0.3.0] — Sprint 1 · C1-1 (2026-09-15)

### C1-1 — ETL + modelo base de no-show
- **ETL** en `app/training/etl.py`: `load_raw` (resolución de rutas contra la raíz del repo), `clean` y `load_cleaned`. Limpieza sobre el dataset de C0-3: descarta `Age<0`, `WaitingDays<0` (6 filas) y binariza `Handcap>0`; excluye la feature endógena `SMS_received`.
- **Modelo base**: pipeline `ColumnTransformer` (StandardScaler + binarias passthrough + OneHotEncoder `handle_unknown="ignore"`) + **RandomForest** con `class_weight='balanced'` (mejor AUC que LogisticRegression sobre el mismo split estratificado).
- **Métricas en test (20%, estratificado):** AUC-ROC **0.7298**, sensibilidad (minoritaria) **0.8313**, especificidad 0.5138.
- **Artefacto** `models/model.joblib` (no versionado, `.gitignore`): pipeline + features + defaults + métricas + versión. `app/training/train.py` expone `build_pipeline()` y `train()`; reentrenar con `python -m app.training.train`.
- **Inferencia alineada** (`predict_service.py`): construye un DataFrame con las features del dataset mapeando `edad→Age`, `genero→Gender`, `dias_espera→WaitingDays`; las columnas que el contrato no expone (`Neighbourhood`, comorbilidades) se rellenan con defaults de entrenamiento (`.joblib`). Eliminado el vector numpy del placeholder.
- **ModelRegistry** valida `model_min_version` (el artefacto debe cumplir `>=0.1.0`); artefacto inválido o obsoleto ⇒ modo degradado.
- `app_version` del contrato actualizada a **1.2.0** (coherente con `ia-api.yaml` v1.2.0); repo a **0.3.0**.
- Feature engineering potencial no incluida en esta iteración (queda documentada): `Neighbourhood` por OneHot (sin target encoding), día de semana no usado, `especialidad`/`ausencias_previas`/`canal_recordatorio` del contrato sin equivalente en el dataset (usan defaults). Decisiones detalladas en `data/CHECKLIST.md`.
- Commit: `422ca98`.

## [0.2.0] — Sprint 0 completo (2026-09-15)

### C0-3 — Selección de dataset (Kaggle)
- **Descargado** `Medical Appointment No Shows` (`data/KaggleV2-May-2016.csv`).
- **Validado:** 110.527 filas × 14 columnas, 0 nulos, licencia **CC BY-NC-SA 4.0**.
- **Variable objetivo** `No-show`: 20,2% → desbalanceo moderado (~1:4).
- **Datos sucios detectados** (pendientes de limpiar en C1-1): `Age=-1` (1), `Handcap>1` (199), `WaitingDays<0` (5).
- **Exclusión de feature:** `SMS_received` (endógeno: el recordatorio se envía al grupo de riesgo).
- **Features propuestas** (C1-1): `WaitingDays` derivada, `Gender`, `Age`, `Neighbourhood` (target encoding), `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`, `Handcap>0`, día de semana.
- **Estrategia de desbalanceo:** `class_weight='balanced'` + métricas AUC-ROC / recall minoritaria (descartar accuracy).
- Añadido `scripts/inspect_dataset.py` (inspección reproducible vía `settings.dataset_path`).
- Añadido `data/CHECKLIST.md` (informe C0-3) — se versiona; el `.csv` queda en `.gitignore`.
- `requirements.txt`: +`pandas`; `.gitignore`: excluye `.coverage` y versiona `CHECKLIST.md`.
- Commit: `d96cfa2`.

## [0.1.0] — Sprint 0 (2026-09-14)

### C0-1 — Inicialización del microservicio (estructura, FastAPI, CI, Docker)
- Estructura modular: `app/`, `tests/`, `scripts/`.
- **FastAPI + Uvicorn + Pydantic v2**, factory `create_app()` (testeable), CORS, lifespan.
- **Configuración** vía `pydantic-settings` + `.env` (`app/config.py`), con `app_version=1.1.1` (alineada al contrato IA).
- **4 endpoints del contrato v1.1.1:**
  - `GET /health`
  - `POST /nlp/sintomas`
  - `POST /nlp/resumen`
  - `POST /predict`
- **Servicios** con inyección manual de dependencias (providers): `ModelRegistry`, `NlpService`, `PredictService`.
- **Modelo placeholder** entrenable (`DummyClassifier`) vía `app.training`; `models/` no versionado.
- **CI (GitHub Actions):** ruff (lint + format) + pytest, matrix Python 3.11/3.12.
- **Docker:** `Dockerfile` (python:3.11-slim) + `docker-compose.yml`.
- **Tests:** 6/6 (health, nlp, predict) con `TestClient`.
- **Requerimientos con rangos flexibles** (compatibilidad Python 3.11 prod vs 3.14 local).
- Bugs corregidos durante la validación: inyección vía `Depends(Clase)` (usar providers) y regex de splitteo de síntomas (clase de caracteres → alternancia).
- Commit: `71ec508`.

---

## Convenciones

- Versiones del **repositorio** (SemVer) en `pyproject.toml`.
- Versión del **contrato API** (`APP_VERSION`) en `app/config.py` — sigue `ia-api.yaml`.
- Los cambios a los **contratos** (repo `pabloordenes/medcare-contracts`) se documentan en el `CHANGELOG.md` de ese repositorio, no aquí.

## Pendiente (roadmap)

| Sprint | Hitos |
|--------|-------|
| Sprint 1 | ~~**C1-1**~~ ETL + modelo base ✅ Delivered · **C1-2** Feature engineering avanzada (target encoding, día de semana) |
| Sprint 2 | **C2-1** `/predict` con modelo cargado en startup ✅ Parcial (startup ya carga) · **C2-2** NLP spaCy/NLTK real |
| Sprint 3 | **C3-1** NLP producción + métricas · Integración `.NET` (`POST /integracion/ml/*`) |
| Sprint 4 | **C4-1** Cron job batch 48h + auth M2M `service-ia` |