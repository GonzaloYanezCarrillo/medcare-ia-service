# Integración demo "Riesgo de Inasistencia (IA)" — frontend

Artefacto de **simulación** (offline) que muestra citas del dataset Kaggle con su
**% de inasistencia real** y la **banda de riesgo** predicha por el modelo.
**No pasa por OMRO ni por .NET**: es un JSON estático que el frontend solo renderiza.

## Archivos a agregar

| Archivo | Destino en el repo frontend |
|---|---|
| `RiesgoIA.tsx` | `src/Pages/Medicos/RiesgoIA/RiesgoIA.tsx` |
| `RiesgoIA.css` | `src/Pages/Medicos/RiesgoIA/RiesgoIA.css` |
| `demo_citas.json` | `public/demo_citas.json` |

## Cambios en `src/App.tsx`

```tsx
import RiesgoIAMedico from './Pages/Medicos/RiesgoIA/RiesgoIA.tsx';
```

Dentro del bloque `<RoleRoute role="medico">`:

```tsx
<Route path="/riesgo-ia-page-medico" element={<RiesgoIAMedico />} />
```

## Cambios en `src/Components/Layouts/Navbar/navConfig.tsx`

```tsx
import { FaHome, FaCalendar, FaBook, FaChartLine } from 'react-icons/fa';
```

En el arreglo `medico`:

```tsx
{ to: '/riesgo-ia-page-medico', label: 'Riesgo IA', icon: FaChartLine },
```

## Aplicar el patch (alternativa)

```bash
git apply demo_riesgo_ia.patch
```

## Regenerar `demo_citas.json` (repo IA)

El JSON lo produce el microservicio IA con el modelo real (v1.3.0):

```powershell
# desde la raíz del repo ia_service
.venv\Scripts\python.exe scripts\generar_demo_citas.py --n 60 --seed 42
# salida: docs/demo/demo_citas.json  ->  copiar a public/demo_citas.json del frontend
```

## Verificación

```bash
npm run build   # tsc -b + vite build
npm run lint    # oxlint
```

## Nota

`demo_citas.json` incluye `"simulacion": true`. Las features `Especialidad`,
`AusenciasPrevias` y `CanalRecordatorio` no existen en el dataset Kaggle y usan
los `defaults` del modelo (por eso el score se apoya en edad / días de espera / día).
