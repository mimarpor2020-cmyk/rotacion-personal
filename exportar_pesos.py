"""Exporta los pesos de modelo/modelo_attrition.keras a modelo/pesos_red.npz (para red_numpy.py).

Se ejecuta UNA vez en tu computador (necesita TensorFlow) y cada vez que reentrenes el modelo:
    python exportar_pesos.py

Al final compara las predicciones de Keras y de NumPy sobre datos de prueba.
"""
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import keras  # noqa: E402

from red_numpy import RedNumpy  # noqa: E402

BASE = os.path.dirname(os.path.abspath(__file__))
ENTRADA = os.path.join(BASE, "modelo", "modelo_attrition.keras")
SALIDA = os.path.join(BASE, "modelo", "pesos_red.npz")


def main():
    modelo = keras.models.load_model(ENTRADA, compile=False)
    pesos, activaciones = {}, []
    for capa in modelo.layers:
        tipo = type(capa).__name__
        if tipo == "Dense":
            W, b = capa.get_weights()
            i = len(activaciones)
            pesos[f"W{i}"], pesos[f"b{i}"] = W, b
            activaciones.append(capa.get_config()["activation"])
        elif tipo in ("Dropout", "InputLayer"):
            continue                       # sin efecto al predecir
        else:
            raise SystemExit(f"Capa {tipo} no soportada por red_numpy.py")
    np.savez(SALIDA, activaciones=np.array(activaciones), **pesos)
    print(f"Guardado {SALIDA}: {len(activaciones)} capas densas {activaciones}")

    # Verificación: misma salida que Keras
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, modelo.input_shape[-1])).astype(np.float32)
    p_keras = modelo.predict(X, verbose=0).ravel()
    p_numpy = RedNumpy(SALIDA).predict(X).ravel()
    dif = np.abs(p_keras - p_numpy).max()
    print(f"Diferencia máxima Keras vs NumPy en 500 casos: {dif:.2e}")
    if dif > 1e-4:
        raise SystemExit("Las predicciones no coinciden: revisa la arquitectura.")
    print("OK: las predicciones coinciden.")


if __name__ == "__main__":
    main()
