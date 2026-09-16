# C0-3 — Selección del Dataset (Medical Appointment No Shows)

**Fecha:** 2026-09-15 · **Owner:** Dev C · **Tarjeta:** C0-3 ✅

## Dataset

- **Nombre:** Medical Appointment No Shows (`KaggleV2-May-2016.csv`)
- **Fuente original (Kaggle):** https://www.kaggle.com/datasets/joniarroba/noshowappointments
- **Autor:** Joni Hoppanen (Aquarela Analytics) — datos del sistema de salud público de Brasil (Vitória, Espiritu Santo).
- **Tamaño:** 10,7 MB · **110.527 filas** · **14 columnas** · 0 nulos.
- **Licencia:** CC BY-NC-SA 4.0 (uso académico/no comercial permitido con atribución). Coherente con el proyecto de título.

## Variables

| # | Columna | Tipo | Descripción |
|---|---------|------|-------------|
| 1 | PatientId | float64 | ID del paciente (anonimizado; originalmente en objeto por pérdida de precisión, se lee como float) |
| 2 | AppointmentID | int64 | ID de la cita |
| 3 | Gender | string | M / F |
| 4 | ScheduledDay | string (ISO 8601) | Fecha/hora de registro de la cita |
| 5 | AppointmentDay | string (ISO 8601) | Fecha de la cita |
| 6 | Age | int64 | Edad |
| 7 | Neighbourhood | string | Barrio del centro de salud (81 niveles) |
| 8 | Scholarship | int64 (0/1) | Beneficiario de Bolsa Família |
| 9 | Hipertension | int64 (0/1) | Diagnóstico de hipertensión |
| 10 | Diabetes | int64 (0/1) | Diagnóstico de diabetes |
| 11 | Alcoholism | int64 (0/1) | Diagnóstico de alcoholismo |
| 12 | Handcap | int64 (0–4) | Número de discapacidades |
| 13 | SMS_received | int64 (0/1) | Recibió recordatorio por SMS |
| 14 | **No-show** | string | **Variable objetivo**: `Yes` no asistió / `No` asistió |

## Validación

- **Filas:** 110.527 ✅ (coincide con el contrato C0-3 del tablero).
- **Nulos:** 0 en todas las columnas ✅.
- **Variable objetivo:** binaria `No-show`, sin ambiguos ✅.
- **Licencia y procedencia:** verificadas; se cita fuente en README/artefacto.

## Datos sucios a limpiar en C1-1 (ETL)

| Problema | Detalle | Tratamiento propuesto |
|---|---|---|
| `Age = -1` | 1 fila con edad negativa | Descartar fila |
| `Age = 115` | Edad máxima extrema (1 fila) | Revisar; si es outlier real retener |
| `Handcap > 1` | Uso histórico: el dataset de competición lo mapea a 1; aquí va 0–4 (Hay 3 filas con 4, 13 con 3) | Binarizar a `>0` (o dejar ordinal, evaluar en EDA) |
| `WaitingDays < 0` | 5 filas donde `AppointmentDay` < `ScheduledDay` (el recordar SMS no puede ser negativo) | Descartar o corregir; nunca negativo en features |
| `SMS_received` endógeno | El recordatorio se envía *porque* existe riesgo de no-show; correlación inversa observada (SMS=1 → 8.9% no-show vs SMS=0 → 11.3%) | **NO usar como feature causal** en C1-1; documentar para decisión posterior |

## Desbalanceo de la variable objetivo

- `No-show`: **22.319 (20,2%)**
- `Show`: **88.208 (79,8%)**

Razón de desbalance ≈ **1:4** → **desbalanceo moderado** (umbral típico de atención < 30%).
Estrategias candidatas para C1-1 (a decidir tras split estratificado):
`class_weight='balanced'` en el modelo, métrica por **AUC-ROC + recall clase minoritaria**
(en vez de accuracy), y evaluar SMOTE solo si el modelo base no alcanza recall razonable.

## Features candidatas (propuesta, a consolidar en C1-1)

1. `WaitingDays` (derivada: `AppointmentDay − ScheduledDay`, truncada a ≥ 0)
2. `Gender` (codificado)
3. `Age`
4. `Neighbourhood` (alta cardinalidad → target encoding o agrupar por frecuencia)
5. `Scholarship`, `Hipertension`, `Diabetes`, `Alcoholism`
6. `Handcap > 0` (binario)
7. Día de la semana de la cita (derivado de `AppointmentDay`)

**Excluida como feature:** `PatientId`, `AppointmentID` (IDs), `ScheduledDay` raw,
`SMS_received` (endógeno, ver arriba).

## Artefactos

- `data/KaggleV2-May-2016.csv` — dataset crudo (NO versionado, en `.gitignore`).
- `scripts/inspect_dataset.py` — inspección reproducible de las validaciones anteriores.
- `models/model.joblib` — artefacto C1-1 (NO versionado, en `.gitignore`).
- Tablero Obsidian: tarjeta C0-3 marcada completa.

---

# C1-1 — ETL + Modelo base (no-show)

**Fecha:** 2026-09-15 · **Owner:** Dev C · **Tarjeta:** C1-1 ✅

## Decisión: entrenar con las features del contrato `PredictRequest`

El modelo se entrena **exclusivamente con los 6 campos definidos en `PredictRequest`**
(`ia-api.yaml` v1.2.0), para alinear entrenamiento e inferencia con lo que el sistema
enviará a `POST /predict`:

| Feature del contrato | Origen de datos | Estado |
|---|---|---|
| `Age` (edad) | Dataset Kaggle | Poblada |
| `Gender` (genero) | Dataset Kaggle (M/F) | Poblada |
| `WaitingDays` (dias_espera) | Derivada en ETL | Poblada |
| `Especialidad` | No existe en el dataset | **NaN → imputer** |
| `AusenciasPrevias` | No existe en el dataset | **NaN → imputer** |
| `CanalRecordatorio` | No existe en el dataset | **NaN → imputer** |

### Cómo se manejan los campos ausentes

- En el ETL (`load_cleaned`), las 3 columnas sin datos se cargan como **NaN**.
- `SimpleImputer(strategy="constant")` en el pipeline las imputa a un valor neutro
  (`"desconocido"` para categóricas, `0` para numéricas) tanto en entrenamiento como en
  inferencia.
- El artefacto guarda `missing_in_training: [Especialidad, AusenciasPrevias, CanalRecordatorio]`
  para trazabilidad.
- **Cuando lleguen datos reales de producción** (reentrenamiento C4-1+): se reentrena con
  el mismo esquema, esas columnas vendrán pobladas y los imputers dejan de intervenir — sin
  cambiar contrato, artefacto ni inferencia.

### Impacto en el modelo

- Con solo `Age`/`Gender`/`WaitingDays` informativas (las otras 3 son constante), las
  comorbilidades y `Neighbourhood` del dataset **no se usan** como features.
- Métricas en test (20%, estratificado): **AUC-ROC 0.7049**, sensibilidad 0.7188,
  especificidad 0.5801 (vs. 0.7298 con el dataset completo en la primera iteración).
- `Handcap` se sigue binarizando a `>0` en `clean` (insumo potencial de reentrenamiento
  futuro, sin impacto en el modelo actual).

### Resultado

```text
model_version: 1.2.0
features: ['Age', 'Gender', 'WaitingDays', 'Especialidad', 'AusenciasPrevias', 'CanalRecordatorio']
missing_in_training: ['Especialidad', 'AusenciasPrevias', 'CanalRecordatorio']
metrics: {auc_roc: 0.7049, sensibilidad: 0.7188, especificidad: 0.5801, n_test: 22105}
```

- Tablero Obsidian: tarjeta C1-1 marcada completa.