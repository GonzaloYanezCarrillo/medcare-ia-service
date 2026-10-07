# Reporte de exploración del modelo de no-show

Artefacto: `models/model.joblib` · versión 1.3.0
Métricas (test): AUC 0.7174 · Sensibilidad 0.7829 · Especificidad 0.549

> Nota (C4-1): desde el retrain con feedback el `model_version` es un **semver propio del
> modelo**, independiente de `app_version` de la API; el artefacto desplegado (1.3.0)
> precede a ese esquema y se renumerará al próximo reentrenamiento promovido.

Consulta base del reporte: `edad=42, genero=F, dias_espera=15, especialidad=medicina_general, ausencias_previas=2, canal_recordatorio=ninguno`

## Ejemplos de consultas

| Escenario | dias_espera | ausencias_previas | Probabilidad | Banda | Clase |
|-----------|-------------|-------------------|--------------|-------|-------|
| Paciente joven, cita rápida | 2 | 0 | 0.520 | Alto | no_asiste |
| Cita promedio | 15 | 2 | 0.600 | Alto | no_asiste |
| Espera larga | 40 | 2 | 0.492 | Medio | asiste |
| Muchas ausencias | 20 | 7 | 0.617 | Alto | no_asiste |
| Peor caso | 60 | 10 | 0.436 | Medio | asiste |

## 01_curva_dias_espera.png
![01_curva_dias_espera.png](graficos/01_curva_dias_espera.png)

## 02_curva_edad.png
![02_curva_edad.png](graficos/02_curva_edad.png)

## 03_curva_ausencias_previas.png
![03_curva_ausencias_previas.png](graficos/03_curva_ausencias_previas.png)

## 04_curva_weekday.png
![04_curva_weekday.png](graficos/04_curva_weekday.png)

## 05_heatmap_waiting_ausencias.png
![05_heatmap_waiting_ausencias.png](graficos/05_heatmap_waiting_ausencias.png)

## 06_bandas_frecuencia.png
![06_bandas_frecuencia.png](graficos/06_bandas_frecuencia.png)

## 07_importancia_features.png
![07_importancia_features.png](graficos/07_importancia_features.png)

## Reentrenamiento con feedback (C4-1 · HU-IA-02)

Flujo: .NET envía el desenlace real vía `POST /feedback/asistencia` → JSONL
(`data/feedback/asistencia.jsonl`) → `python -m app.training.train --feedback` concatena
Kaggle + feedback y aplica el **gate de no-degradación** (AUC y sensibilidad ≥ línea base
− 0.01). El modelo se versiona con semver propio y se archiva en `models/versions/<v>/`
con comparativo en `metrics.json`.

Smoke end-to-end con 40 registros de feedback **sintéticos** (ejecución en directorio
temporal, no afecta al artefacto desplegado):

| Métrica | Baseline 1.0.0 | Retrained 1.1.0 | Delta |
|---------|----------------|-----------------|-------|
| AUC-ROC | 0.7174 | 0.7180 | +0.0006 |
| Sensibilidad | 0.7829 | 0.8103 | +0.0274 |
| Especificidad | 0.5490 | 0.5235 | −0.0255 |

Resultado del gate: **promovido** (sin degradación) → `model_version 1.1.0`, `parent_version
1.0.0`. En producción, con feedback real de mayor volumen, se espera una mejora más marcada
al poblar `Especialidad`, `AusenciasPrevias` y `CanalRecordatorio`.
