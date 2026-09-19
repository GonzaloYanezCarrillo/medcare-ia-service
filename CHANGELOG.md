# Changelog — MedCare IA (`ia_service`)

Todas las modificaciones de este repositorio se documentan aquí, siguiendo [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y [SemVer](https://semver.org/lang/es/).

El servicio implementa el contrato [`ia-api.yaml`](https://github.com/pabloordenes/medcare-contracts) (owner: Dev C).

## [0.3.3] — Sprint 2 · C2-1 (2026-09-18)

### C2-1 — API predict en producción
- **Modelo precargado en startup** (`app/main.py`): el `lifespan` invoca `get_model_registry()` en `asyncio.to_thread`; `/health` refleja la disponibilidad real del artefacto desde el arranque y el primer `/predict` ya lo encuentra en memoria.
- **Dockerfile de producción**: se elimina `--reload` (dev-only), usuario no-root (`app`), `HEALTHCHECK` con stdlib vía `/health`, 1 worker/contendor (escalado por réplicas del orquestador).
- **`.dockerignore`**: excluye `data/`, tests, caches; permite `models/` para el build prod.
- **Estrategia del modelo en despliegue**:
  - Dev: `docker-compose.yml` monta `./models` por bind (se regenera el `.joblib` sin rebuild).
  - Prod: `Dockerfile.prod` + `docker-compose.prod.yml` incorporan el artefacto **dentro de la imagen** (autocontenida, `APP_ENV=production`, sin volumes del host). Documentado en README.
- **Validación Docker en CI** (sin necesidad de Docker local): job `docker-build` genera un artefacto sintético (`scripts/ci_dummy_artifact.py`, versión válida, NUNCA desplegable), construye ambas imágenes y smoke-testea el contenedor prod (`/health` → `healthy` + `model_loaded:true`).
- **Decisión `clase`**: la tarjeta C2-1 pedía "probabilidad + clase + banda", pero el contrato v1.2.0 (`PredictResponse`) solo define `score_riesgo` + `banda_riesgo`. Se **mantiene alineado al contrato**: no se agrega `clase` sin coordinar con Dev B.
- **Tests**: `tests/test_main.py` (precarga en startup y modo degradado sin artefacto). 19 tests en verde.
- Commits: `b290aee` (+Paso 1-3 del plan C2-1).

## [0.3.2] — Deuda de calidad (2026-09-17)

### Tests de `app/training` y higiene del repo
- **Tests** `tests/test_etl.py` y `tests/test_train.py` (traen la cobertura de `app/training` de 0% → 99% y la global a 93%): `clean`/`load_raw`/`load_cleaned` con dataset sintético en `tmp_path` (sin depender del CSV de Kaggle, no versionado en CI); `build_pipeline` ajusta y predice un frame del contrato con las 3 columnas NaN; `_defaults`, `evaluate` y `train` persisten un artefacto joblib completo y verificable. `train()` parchea `load_cleaned` para no tocar `models/model.joblib`.
- **README** actualizado: contrato v1.2.0 (repo `pabloordenes/medcare-contracts`), estado por Sprint C1-1/C1-2 ✅, sección de entrenamiento y nota sobre `WaitingDays`/reentrenamiento en producción.
- **Versión sincronizada**: `app/__init__.py` pasa de 0.1.0 → **0.3.2** (alineada con `pyproject.toml`); `app_version` del contrato sigue en 1.2.0 y `model_min_version` en 0.1.0.
- **Limpieza**: eliminado `.pyc` huérfano de `app/training/features.py` (reemplazado por `etl.py` en C1-1) y `.coverage` obsoleto.
- 17 tests en verde; `ruff check`/`ruff format --check` limpios.

## [0.3.1] — Sprint 1 · C1-2 (2026-09-16)

### C1-2 — Feature engineering avanzada: día de la semana
- **ETL** (`app/training/etl.py`): nueva feature derivada `Weekday` (0=Lunes...6=Domingo) desde `AppointmentDay`; añadida a `FEATURES` como numérica ordinal.
- **Modelo C1-2**: RandomForest re-tunado (`n_estimators=150`, `max_depth=12`, `class_weight="balanced"`).
- **Métricas test (20%, estratificado):** AUC-ROC **0.7174** (vs 0.7049 C1-1), sensibilidad **0.7829** (vs 0.7188), especificidad 0.5490 (vs 0.5801). Mejora AUC y sensibilidad (prioridad de triaje) con leve pérdida de especificidad.
- **Inferencia**: `Weekday` no está en el contrato `PredictRequest` → se rellena con el default de entrenamiento (mediana); sin cambios en `predict_service.py` (el mapeo biunívoco cubre los 6 campos del contrato).
- Commit: `7de2cd7`.

## [0.3.0] — Sprint 1 · C1-1 (2026-09-15)

### C1-1 — ETL + modelo base de no-show
- **ETL** en `app/training/etl.py`: `load_raw` (resolución de rutas contra la raíz del repo), `clean` y `load_cleaned`. Limpieza sobre el dataset de C0-3: descarta `Age<0`, `WaitingDays<0` (6 filas) y binariza `Handcap>0`; excluye la feature endógena `SMS_received`.
- **Alineación con `PredictRequest`**: `load_cleaned` devuelve exactamente los **6 campos del contrato** (`Age`, `Gender`, `WaitingDays`, `Especialidad`, `AusenciasPrevias`, `CanalRecordatorio`). Los 3 sin equivalente en el dataset se cargan como **NaN** y quedan **sin señal** para el modelo mientras no haya datos reales (sklearn las maneja como missing en entrenamiento; en inferencia se rellenan desde `defaults`). El datatest es solo un ejercicio: cuando el modelo se reentrene con datos reales de producción (C4-1+) esas columnas vendrán pobladas y empezarán a aportar. Quedan marcadas en el artefacto (`missing_in_training`).
- **Modelo base**: pipeline `ColumnTransformer` + **RandomForest** con `class_weight='balanced'` y OHE con `handle_unknown="ignore"` sobre las features del contrato (soporta `genero: "Otro"` no visto).
- **Métricas en test (20%, estratificado):** AUC-ROC **0.7049**, sensibilidad (minoritaria) **0.7188**, especificidad 0.5801. Con solo `Age`/`Gender`/`WaitingDays` informativas (las 3 restantes son constante).
- **Artefacto** `models/model.joblib` (no versionado): pipeline + features + `feature_types` + defaults + métricas + versión + `missing_in_training`. Reentrenar con `python -m app.training.train`.
- **Inferencia alineada** (`predict_service.py`): mapeo biunívoco contrato→feature y coerción de dtypes (numéricas float64, categóricas str) para que el pipeline acepte el DataFrame construido desde `defaults`. Eliminado el vector numpy del placeholder.
- **ModelRegistry** valida `model_min_version` (el artefacto debe cumplir `>=0.1.0`); artefacto inválido u obsoleto ⇒ modo degradado.
- `app_version` del contrato actualizada a **1.2.0** (coherente con `ia-api.yaml` v1.2.0); repo a **0.3.0**.
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
| Sprint 1 | ~~**C1-1**~~ ✅ · ~~**C1-2**~~ Feature engineering avanzada (día de semana) ✅ Delivered |
| Sprint 2 | **C2-1** `/predict` con modelo cargado en startup ✅ Parcial (startup ya carga) · **C2-2** NLP spaCy/NLTK real |
| Sprint 3 | **C3-1** NLP producción + métricas · Integración `.NET` (`POST /integracion/ml/*`) |
| Sprint 4 | **C4-1** Cron job batch 48h + auth M2M `service-ia` |