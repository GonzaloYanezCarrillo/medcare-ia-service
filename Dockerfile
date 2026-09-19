# MedCare IA — Imagen del microservicio (FastAPI + Uvicorn)
# Multi-stage implícito: solo se copia lo necesario al runtime (no tests, no artefactos MLOps).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias de runtime primero (mejor caché de capas).
# En local/dev el Dockerfile no instala dev (ruff/pytest): esos viven solo en CI (venv).
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Usuario sin privilegios: si el proceso se compromete, no tiene root en el contenedor.
RUN groupadd --system app && useradd --system --gid app --create-home app
USER app

# Código de la aplicación.
COPY --chown=app:app app ./app

# Accesible para dev y para el healthcheck del contenedor.
EXPOSE 8000

# Healthcheck de los orquestadores: golpea GET /health con la stdlib (sin instalaciones extra).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"]

# Sin --reload: en producción el proceso es de un solo worker por contenedor.
# El escalado horizontal (N réplicas) lo decide el orquestador, no Uvicorn multi-worker.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]