"""Red neuronal en NumPy: misma predicción que el modelo Keras, sin TensorFlow.

El servidor gratuito (Render, 512 MB de RAM) no alcanza para TensorFlow, así que los pesos
entrenados se exportan una vez con `python exportar_pesos.py` a modelo/pesos_red.npz y aquí
se aplica la propagación hacia adelante:

    capa densa:  a = activación(x · W + b)
    Dropout:     en predicción no hace nada (solo actúa durante el entrenamiento)

Arquitectura del notebook: Dense(32, relu) -> Dropout -> Dense(16, relu) -> Dropout -> Dense(1, sigmoid)
"""
import numpy as np

ACTIVACIONES = {
    "relu": lambda z: np.maximum(z, 0.0),
    "sigmoid": lambda z: 1.0 / (1.0 + np.exp(-z)),
    "tanh": np.tanh,
    "linear": lambda z: z,
}


class RedNumpy:
    def __init__(self, ruta_npz):
        datos = np.load(ruta_npz, allow_pickle=False)
        self.activaciones = [str(a) for a in datos["activaciones"]]
        self.capas = [(datos[f"W{i}"].astype(np.float64), datos[f"b{i}"].astype(np.float64))
                      for i in range(len(self.activaciones))]
        for a in self.activaciones:
            if a not in ACTIVACIONES:
                raise ValueError(f"Activación no soportada: {a}")

    def predict(self, X, verbose=0):
        """Misma firma que keras.Model.predict: devuelve un arreglo (n, 1)."""
        a = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)
        for (W, b), act in zip(self.capas, self.activaciones):
            a = ACTIVACIONES[act](a @ W + b)
        return a
