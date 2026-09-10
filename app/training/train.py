"""Entrenamiento del modelo base (esqueleto).

C1-1 implementará el pipeline real:
1. Carga y limpieza del dataset Kaggle (110.527 filas).
2. Ingeniería de features (WaitingDays, especialidad, SMS).
3. División train/test estratificada.
4. Entrenamiento (LogisticRegression / RandomForest).
5. Persistencia del artefacto con joblib y evaluación (sensibilidad/especificidad).

Este módulo expone un placeholder para validar el pipeline de arranque.
"""


def train_placeholder(model_path: str = "models/model.joblib") -> str:
    """Entrena un marcador y lo persiste (placeholder hasta C1-1).

    Returns:
        Ruta del artefacto guardado.
    """
    from pathlib import Path

    import joblib

    # Modelo trivial que respeta la interfaz predict_proba esperada por ModelRegistry.
    import numpy as np
    from sklearn.dummy import DummyClassifier

    x = np.array([[0.0], [1.0]])
    y = np.array([0, 1])
    model = DummyClassifier(strategy="most_frequent").fit(x, y)

    path = Path(model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return str(path)
