# src/core/knn_core.py

from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

import numpy as np


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia euclidiana entre dos vectores 1D.
    """
    diff = a - b
    return float(np.sqrt(np.dot(diff, diff)))


def knn_predict_point(
    test_point: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> int:
    """
    Predice la clase de un solo punto de prueba usando KNN secuencial.
    """
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return int(most_common[0][0])


def knn_predict_batch(
    X_test: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Aplica KNN (versión secuencial) a todos los puntos de prueba.
    """
    preds = [knn_predict_point(x, X_train, y_train, k) for x in X_test]
    return np.array(preds, dtype=int)


def compute_local_neighbors(
    test_point: np.ndarray,
    X_train_chunk: np.ndarray,
    y_train_chunk: np.ndarray,
    k: int,
) -> List[Tuple[float, int]]:
    """
    Calcula los k vecinos más cercanos dentro de un *chunk local* de train.

    Esta función es útil para la versión MPI e híbrida:
    cada proceso llama a compute_local_neighbors sobre su subconjunto
    y luego se hace un gather en el root.
    """
    if X_train_chunk.shape[0] == 0:
        return []

    distances = [euclidean_distance(test_point, x) for x in X_train_chunk]
    k_local = min(k, len(distances))
    local_indices = np.argsort(distances)[:k_local]

    neighbors = [(float(distances[i]), int(y_train_chunk[i])) for i in local_indices]
    return neighbors


def majority_vote(labels: Sequence[int]) -> int:
    """
    Devuelve la etiqueta más frecuente (voto mayoritario).
    """
    if not labels:
        raise ValueError("majority_vote recibió una lista vacía de etiquetas.")
    most_common = Counter(labels).most_common(1)
    return int(most_common[0][0])
