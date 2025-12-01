# src/methods/knn_hybrid.py

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from mpi4py import MPI
from joblib import Parallel, delayed

from core.data_utils import load_digits_data_root
from core.knn_core import compute_local_neighbors, majority_vote
from core.timing_utils import compute_knn_flops


def run_hybrid(
    k: int = 3,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    n_threads: int = 2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Optional[Dict]:
  
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # p

 
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

    n_train_global, n_test_global, n_features = comm.bcast(meta, root=0)

   
    t_comm_local = 0.0
    t_comp_local = 0.0

    t0_comm = MPI.Wtime()
    X_train_local = comm.scatter(X_train_chunks, root=0)
    y_train_local = comm.scatter(y_train_chunks, root=0)

    X_test_local = comm.bcast(X_test_full, root=0)
    y_test_local = comm.bcast(y_test_full, root=0)
    t1_comm = MPI.Wtime()
    t_comm_local += (t1_comm - t0_comm)


    comm.Barrier()
    t_start = MPI.Wtime()

    def local_neighbors_for_test(j: int):
        test_point = X_test_local[j]
        return compute_local_neighbors(
            test_point=test_point,
            X_train_chunk=X_train_local,
            y_train_chunk=y_train_local,
            k=k,
        )

    t0_comp = MPI.Wtime()

    local_neighbors_list = Parallel(
        n_jobs=n_threads,
        prefer="threads",
    )(
        delayed(local_neighbors_for_test)(j)
        for j in range(n_test_global)
    )
    t1_comp = MPI.Wtime()
    t_comp_local += (t1_comp - t0_comp)


    t0_comm = MPI.Wtime()
    all_neighbors_all_procs = comm.gather(local_neighbors_list, root=0)
    t1_comm = MPI.Wtime()
    t_comm_local += (t1_comm - t0_comm)

    predictions = None
    if rank == 0:
   
        predictions = []
        for j in range(n_test_global):
            merged = []
            for p_neighbors in all_neighbors_all_procs:
                merged.extend(p_neighbors[j])

            if merged:
                merged.sort(key=lambda x: x[0])  # ordenar por distancia
                top_k = merged[:k]
                labels = [lab for (_, lab) in top_k]
                pred_label = majority_vote(labels)
            else:
                # Caso límite (no debería pasar en la práctica)
                pred_label = 0

            predictions.append(pred_label)

    comm.Barrier()
    t_end = MPI.Wtime()

 
    t_total_local = t_end - t_start
    t_total = comm.reduce(t_total_local, op=MPI.MAX, root=0)
    t_comp = comm.reduce(t_comp_local, op=MPI.MAX, root=0)
    t_comm = comm.reduce(t_comm_local, op=MPI.MAX, root=0)


    if rank == 0:
        y_pred = np.array(predictions, dtype=int)
        accuracy = float(np.mean(y_pred == y_test_local))

        flops = compute_knn_flops(
            n_train=n_train_global,
            n_test=n_test_global,
            n_features=n_features,
            include_sqrt=True,
        )

        return {
            "p": size,
            "threads": n_threads,
            "n_train": n_train_global,
            "n_test": n_test_global,
            "k": k,
            "accuracy": accuracy,
            "t_total": t_total,
            "t_compute": t_comp,
            "t_comm": t_comm,
            "flops": flops,
        }

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    metrics = run_hybrid(k=3, n_threads=2)

    if rank == 0 and metrics is not None:
        print(
            f"Hybrid KNN | p={metrics['p']}, threads={metrics['threads']}, "
            f"n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
            f"k={metrics['k']}, acc={metrics['accuracy']:.4f}, "
            f"T_total={metrics['t_total']:.4f} s, "
            f"T_comp={metrics['t_compute']:.4f} s, "
            f"T_comm={metrics['t_comm']:.4f} s"
        )
