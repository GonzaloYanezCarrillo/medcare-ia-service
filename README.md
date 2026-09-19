# MedCare IA — Microservicio de Inteligencia Artificial

Microservicio de IA de MedCare AI (Track C, Dev C).
Stack: **Python 3.11 · FastAPI · Uvicorn · Pydantic · Scikit-Learn · spaCy/NLTK · joblib**.

Contrato de referencia: `medcare-contracts/ia-api.yaml` (versión 1.2.0, repo `pabloordenes/medcare-contracts`).

## Estructura del proyecto

```text
ia_service/
├── app/
│   ├── main.py               # App FastAPI (punto de entrada)
│   ├── config.py             # Configuración (pydantic-settings + .env)
│   ├── api/
│   │   └── routers/          # health.py · nlp.py · predict.py
│   ├── schemas/              # Models Pydantic (health.py · nlp.py · predict.py)
│   ├── services/             # Lógica de negocio + model registry
│   └── training/             # ETL, feature engineering y entrenamiento (C1-1/C1-2)
├── models/                   # Artefactos .joblib (no versionado)
├── data/                     # Dataset Kaggle + CHECKLIST (csv no versionado)
├── tests/                    # Pytest + TestClient
├── .github/workflows/ci.yml  # CI: ruff (lint/format) + pytest
├── Dockerfile                # Imagen del servicio
├── docker-compose.yml        # Entorno local con Docker
├── pyproject.toml            # Ruff + pytest
└── requirements*.txt
```

## Endpoints (contrato v1.2.0)

| Método | Ruta            | Descripción                                        |
|--------|-----------------|----------------------------------------------------|
| GET    | `/health`       | Health check (status, modelo cargado, versión)     |
| POST   | `/nlp/sintomas` | Extraer entidades de síntomas desde texto libre    |
| POST   | `/nlp/resumen`  | Resumen clínico preliminar + entidades (HU-NLP-03) |
| POST   | `/predict`      | Score y banda de riesgo de no-show                 |

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
- **Sprint 0 (C0-1, C0-2, C0-3) ✅** estructura, FastAPI+Uvicorn+Pydantic, contrato IA v1.2.0, dataset Kaggle.
- **C1-1 ✅** ETL (`app/training/etl.py`), pipeline RandomForest balanced, artefacto joblib + model registry; features del `PredictRequest` (3 campos aún sin datos → NaN hasta producción).
- **C1-2 ✅** feature `Weekday`, RandomForest retuneado (n=150, depth=12): AUC 0.7174, sensibilidad 0.7829 en test.
- **C2-1 ✅** API predict en producción: modelo precargado en startup, `/health`, Dockerfile prod (sin `--reload`, no-root), imagen prod autocontenida con el artefacto embebido, CI valida ambos builds + smoke test. Respuesta alineada al contrato v1.2.0 (score + banda, sin `clase`).
- **C2-2/C3-1 ▶️** NLP real con spaCy/NLTK (hoy heurística de marcado).

Nota: el modelo actual se entrena con el dataset Kaggle como base de ejercicio; `WaitingDays` se deriva de `ScheduledDay`/`AppointmentDay`. Al pasar a producción se reentrenará con datos reales que poblarán `Especialidad`, `AusenciasPrevias` y `CanalRecordatorio` (sin cambios de código).

## Entrenamiento

```bash
python -m app.training.train
```

Persiste el artefacto en `models/model.joblib` con pipeline, features y métricas.