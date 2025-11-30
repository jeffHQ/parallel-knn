# src/methods/knn_mpi.py

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from mpi4py import MPI

from core.data_utils import load_digits_data_root
from core.knn_core import compute_local_neighbors, majority_vote
from core.timing_utils import compute_knn_flops


def run_mpi(
    k: int = 3,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Optional[Dict]:
    """
    Versión paralela de KNN usando MPI (Message Passing Interface).

    Estrategia de paralelización:
    - El conjunto de entrenamiento se reparte entre procesos (scatter)
    - El conjunto de prueba se replica en todos los procesos (bcast)
    - Cada proceso calcula vecinos locales para todos los test points
    - Se hace gather de vecinos al root para fusión y voto mayoritario

    Parámetros
    ----------
    k : int
        Número de vecinos para clasificación KNN (default: 3).
    n_train : int | None
        Número de muestras de entrenamiento (None = todas disponibles).
    n_test : int | None
        Número de muestras de prueba (None = todas disponibles).
    test_size : float
        Fracción del dataset para test (default: 0.2).
    random_state : int
        Semilla para reproducibilidad del split (default: 42).

    Devuelve
    --------
    dict | None
        En el rank 0: diccionario con métricas (igual estructura que run_sequential,
        más la clave 'p' para número de procesos). En otros ranks: None.
        
        Métricas incluidas:
        - p: número de procesos MPI
        - n_train, n_test, k: parámetros del experimento
        - accuracy: exactitud de clasificación
        - t_total: tiempo total (incluyendo comunicación)
        - t_compute: tiempo de cómputo local (MAX entre procesos)
        - t_comm: tiempo de comunicación (MAX entre procesos)
        - flops: FLOPs teóricos totales
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # p

    # ---------- 1. Carga de datos y particionado (solo root) ----------

    if rank == 0:
        X_train_full, X_test_full, y_train_full, y_test_full = load_digits_data_root(
            n_train=n_train,
            n_test=n_test,
            test_size=test_size,
            random_state=random_state,
        )
        n_train_global = X_train_full.shape[0]
        n_test_global = X_test_full.shape[0]
        n_features = X_train_full.shape[1]

        X_train_chunks = np.array_split(X_train_full, size)
        y_train_chunks = np.array_split(y_train_full, size)

        meta = (n_train_global, n_test_global, n_features)
    else:
        X_train_chunks = None
        y_train_chunks = None
        X_test_full = None
        y_test_full = None
        meta = None

    # Todos saben cuántos train/test globales hay y cuántas features
    n_train_global, n_test_global, n_features = comm.bcast(meta, root=0)

    # ---------- 2. Comunicación inicial (scatter + bcast) ----------

    t_comm_local = 0.0
    t_comp_local = 0.0

    t0_comm = MPI.Wtime()
    # Cada proceso recibe su parte del train
    X_train_local = comm.scatter(X_train_chunks, root=0)
    y_train_local = comm.scatter(y_train_chunks, root=0)
    # Todos reciben copia del test y sus etiquetas
    X_test_local = comm.bcast(X_test_full, root=0)
    y_test_local = comm.bcast(y_test_full, root=0)
    t1_comm = MPI.Wtime()
    t_comm_local += (t1_comm - t0_comm)

    # ---------- 3. Bucle principal de clasificación ----------

    comm.Barrier()
    t_start = MPI.Wtime()

    predictions = []  # solo se llenará en el rank 0

    for j in range(n_test_global):
        test_point = X_test_local[j]

        # --- Cómputo local: distancias a mi chunk de train ---
        t0_comp = MPI.Wtime()
        local_neighbors = compute_local_neighbors(
            test_point=test_point,
            X_train_chunk=X_train_local,
            y_train_chunk=y_train_local,
            k=k,
        )
        t1_comp = MPI.Wtime()
        t_comp_local += (t1_comp - t0_comp)

        # --- Comunicación: gather de vecinos hacia root ---
        t0_comm = MPI.Wtime()
        all_neighbors = comm.gather(local_neighbors, root=0)
        t1_comm = MPI.Wtime()
        t_comm_local += (t1_comm - t0_comm)

        # --- Fusión y voto mayoritario (solo root) ---
        if rank == 0:
            merged = [item for sublist in all_neighbors for item in sublist]
            if merged:
                merged.sort(key=lambda x: x[0])  # ordenar por distancia
                top_k = merged[:k]
                labels = [lab for (_, lab) in top_k]
                pred_label = majority_vote(labels)
            else:
                # Caso extremo: nadie tiene vecinos (no debería ocurrir en práctica)
                pred_label = 0
            predictions.append(pred_label)

    comm.Barrier()
    t_end = MPI.Wtime()

    # ---------- 4. Reducir tiempos (máximo entre ranks) ----------

    t_total_local = t_end - t_start

    t_total = comm.reduce(t_total_local, op=MPI.MAX, root=0)
    t_comp = comm.reduce(t_comp_local, op=MPI.MAX, root=0)
    t_comm = comm.reduce(t_comm_local, op=MPI.MAX, root=0)

    # ---------- 5. Métricas (solo root) ----------

    if rank == 0:
        y_pred = np.array(predictions, dtype=int)
        accuracy = float(np.mean(y_pred == y_test_local))

        # FLOPs totales teóricos (misma fórmula que en la versión secuencial)
        flops = compute_knn_flops(
            n_train=n_train_global,
            n_test=n_test_global,
            n_features=n_features,
            include_sqrt=True,
        )

        metrics: Dict = {
            "p": size,
            "n_train": n_train_global,
            "n_test": n_test_global,
            "k": k,
            "accuracy": accuracy,
            "t_total": t_total,
            "t_compute": t_comp,
            "t_comm": t_comm,
            "flops": flops,
        }
        return metrics

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    m = run_mpi(k=3)

    if rank == 0 and m is not None:
        print(
            f"MPI KNN | p={m['p']}, "
            f"n_train={m['n_train']}, n_test={m['n_test']}, "
            f"k={m['k']}, acc={m['accuracy']:.4f}, "
            f"T_total={m['t_total']:.4f} s, "
            f"T_comp={m['t_compute']:.4f} s, "
            f"T_comm={m['t_comm']:.4f} s"
        )
