# MedCare IA — Imagen del microservicio (FastAPI + Uvicorn)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias primero (mejor caché de capas)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Código de la aplicación
COPY app ./app

# Accesible para dev y para el healthcheck del contenedor
EXPOSE 8000

# Uvicorn con recarga en local; en producción usar uvicorn.workers.TemporalWoker o múltiples workers
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]