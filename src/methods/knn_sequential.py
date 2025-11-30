# src/methods/knn_sequential.py

from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from core.data_utils import load_digits_data
from core.knn_core import knn_predict_batch
from core.timing_utils import wall_time, compute_knn_flops


def run_sequential(
    k: int = 3,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    plot_examples: bool = False,
) -> Dict:
    """
    Ejecuta KNN secuencial sobre el dataset digits, mide tiempo y devuelve métricas.

    Esta es la versión baseline utilizada para calcular speedup y eficiencia
    de las versiones paralelas (MPI, OMP, híbrido).

    Parámetros
    ----------
    k : int
        Número de vecinos para clasificación KNN (default: 3).
    n_train : int | None
        Número de muestras de entrenamiento a usar (None = todas disponibles).
    n_test : int | None
        Número de muestras de prueba a usar (None = todas disponibles).
    plot_examples : bool
        Si es True, muestra algunas predicciones en imágenes usando matplotlib.

    Devuelve
    --------
    dict
        Diccionario con las siguientes claves:
        - n_train: número real de muestras de entrenamiento
        - n_test: número real de muestras de test
        - k: parámetro k usado
        - accuracy: exactitud de clasificación (0.0-1.0)
        - t_total: tiempo total de ejecución en segundos
        - t_compute: tiempo de cómputo (igual a t_total en secuencial)
        - t_comm: tiempo de comunicación (siempre 0.0 en secuencial)
        - flops: FLOPs teóricos totales del cómputo de distancias
    """
    X_train, X_test, y_train, y_test = load_digits_data(
        n_train=n_train,
        n_test=n_test,
    )

    n_tr = X_train.shape[0]
    n_te = X_test.shape[0]
    n_features = X_train.shape[1]

    # Medimos solo el tiempo de cómputo del KNN secuencial
    t0 = wall_time()
    y_pred = knn_predict_batch(X_test, X_train, y_train, k)
    t1 = wall_time()

    t_total = t1 - t0
    t_compute = t_total           # en secuencial, todo es cómputo
    t_comm = 0.0                  # no hay comunicación

    accuracy = float(np.mean(y_pred == y_test))

    flops = compute_knn_flops(
        n_train=n_tr,
        n_test=n_te,
        n_features=n_features,
        include_sqrt=True,
    )

    metrics = {
        "n_train": n_tr,
        "n_test": n_te,
        "k": k,
        "accuracy": accuracy,
        "t_total": t_total,
        "t_compute": t_compute,
        "t_comm": t_comm,
        "flops": flops,
    }

    if plot_examples:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            if i >= len(X_test):
                break
            ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
            ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
            ax.axis("off")
        plt.suptitle("Sample Predictions (Sequential KNN)")
        plt.tight_layout()
        plt.show()

    return metrics


if __name__ == "__main__":
    m = run_sequential(k=3, plot_examples=True)
    print(
        f"Sequential KNN | "
        f"n_train={m['n_train']}, n_test={m['n_test']}, "
        f"k={m['k']}, acc={m['accuracy']:.4f}, "
        f"T_total={m['t_total']:.4f} s"
    )
