# src/experiments/hybrid/experiments_hybrid_weak.py

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from core.data_utils import load_digits_data
from core.csv_utils import append_result_row
from methods.knn_hybrid import run_hybrid


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # p

    parser = argparse.ArgumentParser(
        description="Hybrid Weak Scaling: escalar problema con p*threads"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Número de hilos por proceso en el híbrido (default: 2)",
    )
    parser.add_argument(
        "--base-frac",
        type=float,
        default=0.25,
        help="Fracción base para W=1 en weak scaling (default: 0.25)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Número de vecinos para KNN (default: 3)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Limpiar archivo de resultados antes de empezar (solo con p=1, threads=1)",
    )
    args = parser.parse_args()
    n_threads = args.threads
    base_frac = args.base_frac

    # 1. Tamaño completo del dataset (solo root)
    if rank == 0:
        X_train_full, X_test_full, _, _ = load_digits_data(
            n_train=None,
            n_test=None,
        )
        n_train_full = X_train_full.shape[0]
        n_test_full = X_test_full.shape[0]
    else:
        n_train_full = None
        n_test_full = None

    n_train_full = comm.bcast(n_train_full, root=0)
    n_test_full = comm.bcast(n_test_full, root=0)

    # 2. Weak scaling: W = p * threads
    W = size * n_threads
    frac = base_frac * W
    if frac > 1.0:
        frac = 1.0

    n_train = int(frac * n_train_full)
    n_test = int(frac * n_test_full)

    k = args.k

    out_path = Path("results/hybrid/hybrid_weak.csv")

    # Con --clear y p=1, threads=1 se reinicia el archivo
    if rank == 0 and args.clear and size == 1 and n_threads == 1 and out_path.exists():
        out_path.unlink()
        print(f"[INFO] Archivo {out_path} eliminado.")

    metrics = run_hybrid(
        k=k,
        n_train=n_train,
        n_test=n_test,
        n_threads=n_threads,
    )

    if rank == 0 and metrics is not None:
        extra = {
            "version": "hybrid",
            "scaling": "weak",
            "frac": frac,
            "p": metrics["p"],
            "threads": metrics["threads"],
            "workers": metrics["p"] * metrics["threads"],
        }
        append_result_row(out_path, metrics, extra)

        workload = metrics["n_train"] * metrics["n_test"]
        time_per_pair = metrics["t_total"] / workload

        print(
            f"[HYBRID-WEAK] p={metrics['p']}, threads={metrics['threads']}, "
            f"W={metrics['p'] * metrics['threads']}, frac={frac:.2f}, "
            f"n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
            f"k={metrics['k']}, acc={metrics['accuracy']:.4f}, "
            f"T_total={metrics['t_total']:.6f} s, workload={workload}, "
            f"time_per_pair={time_per_pair:.3e} s"
        )


if __name__ == "__main__":
    main()
