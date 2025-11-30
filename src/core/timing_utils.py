# src/core/timing_utils.py

from __future__ import annotations

import time
from typing import Optional


def wall_time() -> float:
    """
    Devuelve tiempo de pared de alta resolución.

    Úsalo en código secuencial / OMP. En MPI sigue siendo mejor usar
    MPI.Wtime() directamente dentro de los métodos MPI.
    """
    return time.perf_counter()


def compute_knn_flops(
    n_train: int,
    n_test: int,
    n_features: int,
    include_sqrt: bool = True,
) -> float:
    """
    Estima el número total de FLOPs de la región de cómputo de distancias
    en KNN (todas las distancias train-test).

    Modelo aproximado por distancia:
      - d restas           -> d FLOPs
      - d multiplicaciones -> d FLOPs
      - d-1 sumas          -> ~d FLOPs
      - sqrt               -> ~1 FLOP (opcional)

    Total aprox: 3*d (+1 si se incluye la raíz cuadrada).
    """
    flops_per_distance = 3 * n_features + (1 if include_sqrt else 0)
    return float(n_train) * float(n_test) * float(flops_per_distance)


def compute_speedup(T_seq: float, T_parallel: float) -> Optional[float]:
    """
    Speedup clásico: S = T_seq / T_parallel.
    Devuelve None si T_parallel es 0 (para evitar división por cero).
    """
    if T_parallel <= 0.0:
        return None
    return T_seq / T_parallel


def compute_efficiency(speedup: float, workers: int) -> Optional[float]:
    """
    Eficiencia paralela: E = speedup / workers.
    """
    if workers <= 0:
        return None
    return speedup / workers
