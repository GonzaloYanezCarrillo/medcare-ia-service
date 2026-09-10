"""Ingeniería de features reutilizable por entrenamiento e inferencia.

Las features replican el contrato de `POST /predict`: edad, género, días de espera,
especialidad, ausencias previas y canal de recordatorio.
"""


def build_feature_vector(
    *,
    edad: int,
    genero: str,
    dias_espera: int,
    especialidad: str,
    ausencias_previas: int,
    canal_recordatorio: str = "ninguno",
) -> dict:
    """Devuelve el vector de features en formato normalizado para el modelo."""
    return {
        "edad": edad,
        "genero": genero,
        "dias_espera": dias_espera,
        "especialidad": especialidad,
        "ausencias_previas": ausencias_previas,
        "canal_recordatorio": canal_recordatorio,
    }
