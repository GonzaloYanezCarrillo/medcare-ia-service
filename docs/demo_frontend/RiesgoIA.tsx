import { useEffect, useMemo, useState } from 'react';
import './RiesgoIA.css';

type Banda = 'Bajo' | 'Medio' | 'Alto';
type FiltroBanda = 'Todas' | Banda;

interface CitaDemo {
    cita_id: string;
    edad: number;
    genero: string;
    dias_espera: number;
    weekday: number;
    dia_semana: string;
    inasistencia_real: number;
    score_riesgo: number;
    banda_riesgo: Banda;
    clase: 'asiste' | 'no_asiste';
}

interface ResumenDemo {
    n_citas: number;
    tasa_inasistencia_real: number;
    tasa_inasistencia_predicha: number;
    auc_muestra: number | null;
    por_banda: Record<Banda, number>;
    por_clase: Record<string, number>;
}

interface DemoCitas {
    generado_en: string;
    simulacion: boolean;
    modelo: { version: string; auc_test: number; sensibilidad: number; especificidad: number };
    umbrales_banda: Record<Banda, string>;
    resumen: ResumenDemo;
    citas: CitaDemo[];
}

const BANDAS: Banda[] = ['Bajo', 'Medio', 'Alto'];
const FILTROS: FiltroBanda[] = ['Todas', ...BANDAS];

const formatearPct = (valor: number) => `${(valor * 100).toFixed(1)}%`;

function RiesgoIAMedico() {
    const [data, setData] = useState<DemoCitas | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [cargando, setCargando] = useState(true);
    const [filtro, setFiltro] = useState<FiltroBanda>('Todas');

    useEffect(() => {
        let activo = true;
        fetch(`${import.meta.env.BASE_URL}demo_citas.json`)
            .then((res) => {
                if (!res.ok) throw new Error(`No se pudo cargar demo_citas.json (${res.status})`);
                return res.json();
            })
            .then((json: DemoCitas) => {
                if (activo) {
                    setData(json);
                    setError(null);
                }
            })
            .catch((err: unknown) => {
                if (activo) setError(err instanceof Error ? err.message : 'Error desconocido');
            })
            .finally(() => {
                if (activo) setCargando(false);
            });
        return () => {
            activo = false;
        };
    }, []);

    const citasFiltradas = useMemo(() => {
        const citas = data?.citas ?? [];
        return filtro === 'Todas' ? citas : citas.filter((c) => c.banda_riesgo === filtro);
    }, [data, filtro]);

    if (cargando) {
        return (
            <div className="ContenedorPrincipal-RiesgoIA">
                <p className="Estado-RiesgoIA">Cargando datos de demostración...</p>
            </div>
        );
    }

    if (error || !data) {
        return (
            <div className="ContenedorPrincipal-RiesgoIA">
                <p className="Estado-RiesgoIA Error">{error ?? 'Sin datos de demostración'}</p>
            </div>
        );
    }

    const { resumen, modelo } = data;

    return (
        <div className="ContenedorPrincipal-RiesgoIA">
            <div className="ContenedorTitulo-RiesgoIA">
                <h1>Riesgo de Inasistencia (IA)</h1>
                <span className="Badge-Simulacion-RiesgoIA">Simulación</span>
            </div>

            <div className="ContenedorContenido-RiesgoIA">
                <p className="Meta-RiesgoIA">
                    Modelo v{modelo.version} · AUC test {modelo.auc_test} · muestra de {resumen.n_citas}{' '}
                    citas del dataset Kaggle (artefacto offline: no pasa por OMRO ni por .NET)
                </p>

                <div className="Kpis-RiesgoIA">
                    <div className="Kpi-RiesgoIA">
                        <span className="KpiValor-RiesgoIA">{resumen.n_citas}</span>
                        <span className="KpiEtiqueta-RiesgoIA">Citas evaluadas</span>
                    </div>
                    <div className="Kpi-RiesgoIA">
                        <span className="KpiValor-RiesgoIA">
                            {formatearPct(resumen.tasa_inasistencia_real)}
                        </span>
                        <span className="KpiEtiqueta-RiesgoIA">Inasistencia real</span>
                    </div>
                    <div className="Kpi-RiesgoIA">
                        <span className="KpiValor-RiesgoIA">
                            {formatearPct(resumen.tasa_inasistencia_predicha)}
                        </span>
                        <span className="KpiDescripcion-RiesgoIA">Riesgo medio predicho</span>
                    </div>
                    <div className="Kpi-RiesgoIA">
                        <span className="KpiValor-RiesgoIA">
                            {resumen.auc_muestra !== null ? resumen.auc_muestra : '—'}
                        </span>
                        <span className="KpiDescripcion-RiesgoIA">AUC en la muestra</span>
                    </div>
                </div>

                <div className="Bandas-RiesgoIA">
                    {BANDAS.map((banda) => (
                        <div key={banda} className={`Chip-Banda-RiesgoIA ${banda}`}>
                            <span className="ChipCantidad-RiesgoIA">{resumen.por_banda[banda]}</span>
                            <span className="ChipEtiqueta-RiesgoIA">Riesgo {banda}</span>
                        </div>
                    ))}
                </div>

                <div className="Filtros-RiesgoIA">
                    {FILTROS.map((opcion) => (
                        <button
                            key={opcion}
                            type="button"
                            className={`BtnFiltro-RiesgoIA ${filtro === opcion ? 'Activo' : ''}`}
                            onClick={() => setFiltro(opcion)}
                        >
                            {opcion}
                        </button>
                    ))}
                </div>

                <div className="TablaWrapper-RiesgoIA">
                    <table className="Tabla-RiesgoIA">
                        <thead>
                            <tr>
                                <th>Cita</th>
                                <th>Edad</th>
                                <th>Género</th>
                                <th>Días de espera</th>
                                <th>Día</th>
                                <th>Inasistió</th>
                                <th>Riesgo predicho</th>
                                <th>Banda</th>
                            </tr>
                        </thead>
                        <tbody>
                            {citasFiltradas.length === 0 ? (
                                <tr>
                                    <td colSpan={8} className="TablaVacia-RiesgoIA">
                                        No hay citas en esta banda
                                    </td>
                                </tr>
                            ) : (
                                citasFiltradas.map((cita) => (
                                    <tr key={cita.cita_id}>
                                        <td className="CeldaId-RiesgoIA">{cita.cita_id}</td>
                                        <td>{cita.edad}</td>
                                        <td>{cita.genero}</td>
                                        <td>{cita.dias_espera}</td>
                                        <td>{cita.dia_semana}</td>
                                        <td>
                                            <span
                                                className={`EtiquetaReal-RiesgoIA ${
                                                    cita.inasistencia_real ? 'Si' : 'No'
                                                }`}
                                            >
                                                {cita.inasistencia_real ? 'Sí' : 'No'}
                                            </span>
                                        </td>
                                        <td>
                                            <div className="BarraRiesgo-RiesgoIA">
                                                <div
                                                    className={`BarraRiesgoRelleno-RiesgoIA ${cita.banda_riesgo}`}
                                                    style={{ width: `${Math.round(cita.score_riesgo * 100)}%` }}
                                                />
                                                <span className="BarraRiesgoValor-RiesgoIA">
                                                    {formatearPct(cita.score_riesgo)}
                                                </span>
                                            </div>
                                        </td>
                                        <td>
                                            <span className={`Pastilla-RiesgoIA ${cita.banda_riesgo}`}>
                                                {cita.banda_riesgo}
                                            </span>
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}

export default RiesgoIAMedico;
