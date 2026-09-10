# MedCare IA — Microservicio de Inteligencia Artificial

Microservicio de IA de MedCare AI (Track C, Dev C).
Stack: **Python 3.11 · FastAPI · Uvicorn · Pydantic · Scikit-Learn · spaCy/NLTK · joblib**.

Contrato de referencia: `contracts/ia-api.yaml` (versión 1.1.1).

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
│   └── training/             # ETL, features y entrenamiento del modelo (C1-1)
├── models/                   # Artefactos .joblib (no versionado)
├── data/                     # Dataset Kaggle (no versionado)
├── tests/                    # Pytest + TestClient
├── .github/workflows/ci.yml  # CI: ruff (lint/format) + pytest
├── Dockerfile                # Imagen del servicio
├── docker-compose.yml        # Entorno local con Docker
├── pyproject.toml            # Ruff + pytest
└── requirements*.txt
```

## Endpoints (contrato v1.1.1)

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
docker compose up --build
```

## Lint y tests

```bash
ruff check .
ruff format --check .
pytest --cov=app
```

## Estado por Sprint
- **C0-1 ✅** estructura, FastAPI+Uvicorn+Pydantic, Dockerfile, CI (ruff+pytest).
- **C0-2 ✅** contrato IA completo (nlp/sintomas, nlp/resumen, predict, health).
- **C0-3/C1-1 ▶️** dataset Kaggle, pipeline ETL y modelo real (placeholder por ahora).
- **C2-2/C3-1 ▶️** NLP real con spaCy/NLTK (hoy heurística de marcado).