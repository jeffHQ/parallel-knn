# knn_omp.py
from collections import Counter
import time

import numpy as np
from joblib import Parallel, delayed

from knn_sequential import load_data, euclidean_distance


def knn_predict_point(test_point: np.ndarray,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      k: int) -> int:
    """
    Predice la clase de un punto de prueba usando el mismo KNN
    que en la versión secuencial.
    """
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def knn_predict_batch_omp(X_test: np.ndarray,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          k: int,
                          n_threads: int) -> np.ndarray:
    """
    Aplica KNN a todos los puntos de prueba usando paralelismo
    de memoria compartida (joblib) sobre los puntos de prueba.
    """
    preds = Parallel(n_jobs=n_threads, prefer="threads")(
    delayed(knn_predict_point)(x, X_train, y_train, k)
    for x in X_test
    )


    return np.array(preds)


def run_omp(k: int = 3,
            n_train: int | None = None,
            n_test: int | None = None,
            n_threads: int = 1) -> dict:
    """
    Ejecuta KNN paralelo tipo OMP (joblib) y devuelve métricas.
    """
    X_train, X_test, y_train, y_test = load_data(
        n_train=n_train,
        n_test=n_test,
    )

    n_tr, n_te = X_train.shape[0], X_test.shape[0]

    start_time = time.time()
    y_pred = knn_predict_batch_omp(
        X_test, X_train, y_train, k, n_threads=n_threads
    )
    end_time = time.time()

    accuracy = float(np.mean(y_pred == y_test))
    t_total = end_time - start_time

    metrics = {
        "threads": n_threads,
        "n_train": n_tr,
        "n_test": n_te,
        "k": k,
        "accuracy": accuracy,
        "t_total": t_total,
    }
    return metrics


if __name__ == "__main__":
    m = run_omp(k=3, n_threads=4)
    print(
        f"OMP KNN | threads={m['threads']}, "
        f"n_train={m['n_train']}, n_test={m['n_test']}, "
        f"k={m['k']}, acc={m['accuracy']:.4f}, "
        f"time={m['t_total']:.4f} s"
    )
