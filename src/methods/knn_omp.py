# src/methods/knn_omp.py

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from joblib import Parallel, delayed

from core.data_utils import load_digits_data
from core.knn_core import knn_predict_point
from core.timing_utils import wall_time, compute_knn_flops


def knn_predict_batch_omp(
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    n_threads: int,
) -> np.ndarray:
    """
    Aplica KNN a todos los puntos de prueba usando paralelismo
    de memoria compartida (joblib) sobre los puntos de prueba.
    """
    preds = Parallel(n_jobs=n_threads, prefer="threads")(
        delayed(knn_predict_point)(x, X_train, y_train, k) for x in X_test
    )
    return np.array(preds, dtype=int)


def run_omp(
    k: int = 3,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_threads: int = 1,
) -> Dict:
    """
    Ejecuta KNN paralelo usando threads de memoria compartida (estilo OpenMP).

    Implementado con joblib (prefer="threads") para paralelizar sobre los
    puntos de test. Cada thread clasifica un subconjunto de test points
    usando todo el conjunto de entrenamiento (compartido en memoria).

    NOTA: En Python, debido al GIL (Global Interpreter Lock), el paralelismo
    con threads puro no escala tan bien como MPI. Sin embargo, operaciones
    NumPy vectorizadas pueden liberar el GIL parcialmente.

    Parámetros
    ----------
    k : int
        Número de vecinos para clasificación KNN (default: 3).
    n_train : int | None
        Número de muestras de entrenamiento (None = todas disponibles).
    n_test : int | None
        Número de muestras de prueba (None = todas disponibles).
    n_threads : int
        Número de hilos (n_jobs de joblib, default: 1).

    Devuelve
    --------
    dict
        Diccionario con métricas:
        - threads: número de hilos usados
        - n_train, n_test, k: parámetros del experimento
        - accuracy: exactitud de clasificación
        - t_total: tiempo total de ejecución
        - t_compute: tiempo de cómputo (igual a t_total)
        - t_comm: tiempo de comunicación (siempre 0.0, no hay comunicación explícita)
        - flops: FLOPs teóricos totales
    """
    X_train, X_test, y_train, y_test = load_digits_data(
        n_train=n_train,
        n_test=n_test,
    )

    n_tr = X_train.shape[0]
    n_te = X_test.shape[0]
    n_features = X_train.shape[1]

    t0 = wall_time()
    y_pred = knn_predict_batch_omp(
        X_test=X_test,
        X_train=X_train,
        y_train=y_train,
        k=k,
        n_threads=n_threads,
    )
    t1 = wall_time()

    t_total = t1 - t0
    t_compute = t_total      # no hay comunicación explícita
    t_comm = 0.0

    accuracy = float(np.mean(y_pred == y_test))

    flops = compute_knn_flops(
        n_train=n_tr,
        n_test=n_te,
        n_features=n_features,
        include_sqrt=True,
    )

    metrics: Dict = {
        "threads": n_threads,
        "n_train": n_tr,
        "n_test": n_te,
        "k": k,
        "accuracy": accuracy,
        "t_total": t_total,
        "t_compute": t_compute,
        "t_comm": t_comm,
        "flops": flops,
    }
    return metrics


if __name__ == "__main__":
    m = run_omp(k=3, n_threads=4)
    print(
        f"OMP KNN | threads={m['threads']}, "
        f"n_train={m['n_train']}, n_test={m['n_test']}, "
        f"k={m['k']}, acc={m['accuracy']:.4f}, "
        f"T_total={m['t_total']:.4f} s"
    )
