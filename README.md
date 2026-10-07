# MedCare IA — Microservicio de Inteligencia Artificial

Microservicio de IA de MedCare AI (Track C, Dev C).
Stack: **Python 3.11 · FastAPI · Uvicorn · Pydantic · Scikit-Learn · spaCy/NLTK · joblib**.

Contrato de referencia: `medcare-contracts/ia-api.yaml` (versión 1.4.0, repo `pabloordenes/medcare-contracts`).

## Estructura del proyecto

```text
ia_service/
├── app/
│   ├── main.py               # App FastAPI (punto de entrada)
│   ├── config.py             # Configuración (pydantic-settings + .env)
│   ├── api/
│   │   └── routers/          # health.py · nlp.py · predict.py · feedback.py
│   ├── schemas/              # Models Pydantic (health · nlp · predict · feedback)
│   ├── services/             # Lógica de negocio + model registry + feedback store
│   └── training/             # ETL, feature engineering, entrenamiento y retrain (C1-1/C1-2/C4-1)
├── models/                   # Artefactos .joblib + versions/ y latest.json (no versionado)
├── data/                     # Dataset Kaggle + CHECKLIST (csv no versionado)
├── tests/                    # Pytest + TestClient
├── .github/workflows/ci.yml  # CI: ruff (lint/format) + pytest
├── Dockerfile                # Imagen del servicio
├── docker-compose.yml        # Entorno local con Docker
├── pyproject.toml            # Ruff + pytest
└── requirements*.txt
```

## Endpoints (contrato v1.4.0)

| Método | Ruta                  | Auth M2M  | Descripción                                        |
|--------|-----------------------|-----------|----------------------------------------------------|
| GET    | `/health`             | Pública   | Health check (status, modelo cargado, versión)     |
| POST   | `/nlp/sintomas`       | `service-ia` | Extraer entidades de síntomas desde texto libre  |
| POST   | `/nlp/resumen`        | `service-ia` | Resumen clínico preliminar + entidades (HU-NLP-03) |
| POST   | `/predict`            | `service-ia` | Score, banda y clase de riesgo de no-show         |
| POST   | `/feedback/asistencia`| `service-ia` | Registrar asistencia real de una cita (HU-IA-02)  |

El contrato `ia-api.yaml` declara `bearerAuth` global. Todas las rutas excepto
`/health` exigen un **JWT M2M** con rol `service-ia` (firma RS256 verificada contra el
JWKS de Supabase en producción, o contra `JWT_PUBLIC_KEY_PEM` en dev/test). Sin auth
eso se traduce en `401`/`403`; si el servicio no tiene ninguna fuente de claves
configurada (`JWT_JWKS_URL`/`JWT_PUBLIC_KEY_PEM`) responde `503` (deniega en lugar de
abrir las rutas).

Para probar localmente sin Supabase, generar un token RS256 firmado con la privada
correspondiente a `JWT_PUBLIC_KEY_PEM` y pasar `Authorization: Bearer <jwt>`:
`iss`/`aud` deben coincidir con `JWT_ISSUER`/`JWT_AUDIENCE` y `role` = `service-ia`.

Swagger interactivo: `http://localhost:8000/docs`.

## Ejecución local (venv)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

## Ejecución con Docker

```bash
# Dev: monta ./app y ./models por volumen (bind). El modelo se regenera sin reconstruir.
docker compose up --build
```

## Despliegue de producción (Docker)

El modelo (`.joblib`, 40 MB) no se versiona. Estrategia: **imagen autocontenida** que
embebe el artefacto en el build.

```bash
# 1. Entrenar (o usar el artefacto existente): genera models/model.joblib
python -m app.training.train

# 2. Construir la imagen prod (copia el artefacto dentro de la imagen)
docker build -f Dockerfile.prod -t medcare-ia:prod .

# 3. Levantar sin volumes del host
docker compose -f docker-compose.prod.yml up -d
```

- Dev (`docker-compose.yml`): el modelo vive en `./models/` del host y se monta por bind.
- Prod (`Dockerfile.prod` + `docker-compose.prod.yml`): el modelo vive **dentro** de la
  imagen; `APP_ENV=production`; sin `--reload`; usuario no-root; healthcheck vía `/health`.

## Lint y tests

```bash
ruff check .
ruff format --check .
pytest --cov=app
```

## Estado por Sprint
- **Sprint 0 (C0-1, C0-2, C0-3) ✅** estructura, FastAPI+Uvicorn+Pydantic, contrato IA v1.3.0, dataset Kaggle.
- **C1-1 ✅** ETL (`app/training/etl.py`), pipeline RandomForest balanced, artefacto joblib + model registry; features del `PredictRequest` (3 campos aún sin datos → NaN hasta producción).
- **C1-2 ✅** feature `Weekday`, RandomForest retuneado (n=150, depth=12): AUC 0.7174, sensibilidad 0.7829 en test.
- **C2-1 ✅** API predict en producción: modelo precargado en startup, `/health`, Dockerfile prod (sin `--reload`, no-root), imagen prod autocontenida con el artefacto embebido, CI valida ambos builds + smoke test. Respuesta alineada al contrato v1.3.0: `score_riesgo`, `banda_riesgo` y `clase` (predicción discreta; umbral 0.5 → `no_asiste`).
- **C2-2 ✅** NLP con spaCy (`es_core_news_sm`) integrado en `nlp_service` como motor de análisis: carga diferida y cacheada, y matching de severidad/urgencia por **lemas** (detecta variaciones morfológicas que la heurística cruda no veía, p.ej. `intensos`, `convulsiones`). Los endpoints `/nlp/*` mantienen el contrato v1.3.0. Extracción fina de entidades y resumen clínico → **C3-1**.
- **C5-1 ✅ (impl.)** Auth M2M: `app/services/auth.py` protege `/predict` y `/nlp/*` con JWT RS256 y rol `service-ia` (JWKS de Supabase en prod, `JWT_PUBLIC_KEY_PEM` en dev/test; fail-closed → `503` si no hay claves configuradas). Rutas sin token → `401`; rol incorrecto → `403`; `/health` sigue pública. Tablero lo mantiene en Sprint 5 (no se re-planificó).
- **C3-1 ✅** Resumen clínico preliminar por cita: `POST /nlp/resumen` devuelve una ficha estructurada (`Ficha clínica preliminar — Cita {id}`) con Síntomas, Duración, Medicamentos y Alergias detectados, más Urgencia sugerida. Se extraen entidades `medicamento` y `alergia` (ya declaradas en el tipo del contrato) sin romper el schema HTTP. Resta (post-MVP): integrar contexto de cita real y validación clínica.
- **C4-1 ✅ (impl.)** Feedback y retrain (HU-IA-02): `POST /feedback/asistencia` (rol `service-ia`) persiste el desenlace real en `data/feedback/asistencia.jsonl` (upsert idempotente por `cita_id`, whitelist RNF-SEG-03: solo features + etiqueta). `python -m app.training.train --feedback` concatena el feedback al dataset Kaggle, aplica un **gate de no-degradación** (AUC y sensibilidad ≥ línea base − 0.01) y, solo si promueve, versiona el modelo con **semver propio** (independiente de `app_version`; bump menor con feedback, patch en reentrenamientos) archivándolo en `models/versions/<v>/` con puntero `latest.json` y comparativo `metrics.json`. Recarga del modelo: **offline + reinicio** (sin hot-reload).

Nota: el modelo actual se entrena con el dataset Kaggle como base de ejercicio; `WaitingDays` se deriva de `ScheduledDay`/`AppointmentDay`. Al pasar a producción se reentrenará con datos reales que poblarán `Especialidad`, `AusenciasPrevias` y `CanalRecordatorio` (sin cambios de código).

## Entrenamiento

```bash
# Modelo base (solo dataset Kaggle)
python -m app.training.train

# Reentrenamiento con feedback real (HU-IA-02) + gate de no-degradación
python -m app.training.train --feedback
```

Persiste el artefacto en `models/model.joblib` con pipeline, features y métricas, y además:

- **Versión del modelo** (`model_version`): semver propio del modelo, **independiente** de
  `app_version` de la API. Sin artefacto previo → `1.0.0`; reentrenamiento con feedback
  (datos nuevos) → bump menor; reentrenamiento del mismo dataset → bump patch.
- **Linaje**: `parent_version` y `trained_at` en el artefacto; copia archivada en
  `models/versions/<v>/model.joblib` y puntero `models/latest.json`.
- **Gate de no-degradación** (solo con `--feedback`): el candidato se promueve si
  AUC-ROC y sensibilidad no caen más de `--epsilon` (default `0.01`) por debajo del
  artefacto vigente. Si falla, **no** se sobrescribe el modelo activo y el CLI sale con
  código `1`.
- **Comparativo** `models/metrics.json`: baseline vs. reentrenado (métricas y delta) y
  resultado del gate, también cuando el candidato es rechazado (auditoría).

El feedback se envía desde .NET con `POST /feedback/asistencia` (rol `service-ia`) y se
almacena en `data/feedback/asistencia.jsonl` (upsert por `cita_id`). La recarga del modelo
en el servicio es **offline + reinicio** (sin endpoint de hot-reload).