# Reporte de exploración del modelo de no-show

Artefacto: `models/model.joblib` · versión 1.3.0
Métricas (test): AUC 0.7174 · Sensibilidad 0.7829 · Especificidad 0.549

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
