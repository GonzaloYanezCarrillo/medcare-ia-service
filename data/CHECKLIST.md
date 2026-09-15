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
- Tablero Obsidian: tarjeta C0-3 marcada completa.