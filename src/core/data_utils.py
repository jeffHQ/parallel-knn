# src/core/data_utils.py

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def load_digits_data(
    test_size: float = 0.2,
    random_state: int = 42,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga el dataset digits, hace train/test split estratificado y
    opcionalmente recorta el número de muestras de train y test.

    Se usa tanto en la versión secuencial como en OMP/MPI/Híbrido.
    """
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=test_size,
        random_state=random_state,
        stratify=digits.target,
    )

    if n_train is not None:
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

    if n_test is not None:
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]

    return X_train, X_test, y_train, y_test


def load_digits_data_root(
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Versión pensada para el rank 0 en MPI.

    Hace exactamente lo mismo que load_digits_data, pero la semántica es:
    'solo la ejecuta el root y luego reparte/broadcastea lo que necesita'.
    """
    return load_digits_data(
        test_size=test_size,
        random_state=random_state,
        n_train=n_train,
        n_test=n_test,
    )


def get_digits_feature_dim() -> int:
    """
    Devuelve el número de características (dimensión) del dataset digits.
    Útil para calcular FLOPs teóricos.
    """
    digits = load_digits()
    return digits.data.shape[1]
