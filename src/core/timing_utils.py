# src/core/timing_utils.py

from __future__ import annotations

import time
from typing import Optional


def wall_time() -> float:
   
    return time.perf_counter()


def compute_knn_flops(
    n_train: int,
    n_test: int,
    n_features: int,
    include_sqrt: bool = True,
) -> float:
    
    flops_per_distance = 3 * n_features + (1 if include_sqrt else 0)
    return float(n_train) * float(n_test) * float(flops_per_distance)


def compute_speedup(T_seq: float, T_parallel: float) -> Optional[float]:
   
    if T_parallel <= 0.0:
        return None
    return T_seq / T_parallel


def compute_efficiency(speedup: float, workers: int) -> Optional[float]:
    
    if workers <= 0:
        return None
    return speedup / workers
