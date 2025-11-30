# src/experiments/hybrid/experiments_hybrid_strong.py

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
        description="Hybrid Strong Scaling: problema fijo, variar p*threads"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Número de hilos por proceso en el híbrido (default: 2)",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Fracciones del dataset a usar (default: 0.25 0.5 0.75 1.0)",
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

    fractions = args.fractions
    k = args.k

    out_path = Path("results/hybrid/hybrid_strong.csv")

    # Con --clear y p=1, threads=1 se reinicia el archivo
    if rank == 0 and args.clear and size == 1 and n_threads == 1 and out_path.exists():
        out_path.unlink()
        print(f"[INFO] Archivo {out_path} eliminado.")

    for frac in fractions:
        n_train = int(frac * n_train_full)
        n_test = int(frac * n_test_full)

        metrics = run_hybrid(
            k=k,
            n_train=n_train,
            n_test=n_test,
            n_threads=n_threads,
        )

        if rank == 0 and metrics is not None:
            extra = {
                "version": "hybrid",
                "scaling": "strong",
                "frac": frac,
                "p": metrics["p"],
                "threads": metrics["threads"],
                "workers": metrics["p"] * metrics["threads"],
            }
            append_result_row(out_path, metrics, extra)

            print(
                f"[HYBRID-STRONG] p={metrics['p']}, threads={metrics['threads']}, "
                f"frac={frac:.2f}, n_train={metrics['n_train']}, "
                f"n_test={metrics['n_test']}, acc={metrics['accuracy']:.4f}, "
                f"T_total={metrics['t_total']:.6f} s"
            )


if __name__ == "__main__":
    main()
