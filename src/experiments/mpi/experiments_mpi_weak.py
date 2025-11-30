# src/experiments/mpi/experiments_mpi_weak.py

from __future__ import annotations

import argparse
from pathlib import Path

from mpi4py import MPI

from core.data_utils import load_digits_data
from core.csv_utils import append_result_row
from methods.knn_mpi import run_mpi


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # p

    parser = argparse.ArgumentParser(
        description="MPI Weak Scaling: escalar problema con procesos"
    )
    parser.add_argument(
        "--base-frac",
        type=float,
        default=0.25,
        help="Fracción base para p=1 (default: 0.25)",
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
        help="Limpiar archivo de resultados antes de empezar (solo con p=1)",
    )
    args = parser.parse_args()

    # 1. Obtener tamaño completo del dataset (solo root)
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

    # 2. Definir fracción según p (weak scaling simple)
    base_frac = args.base_frac
    frac = base_frac * size
    if frac > 1.0:
        frac = 1.0

    n_train = int(frac * n_train_full)
    n_test = int(frac * n_test_full)

    k = args.k

    out_path = Path("results/mpi/mpi_weak.csv")

    # Solo en p=1 con --clear borramos el archivo para reiniciar experimento weak
    if rank == 0 and args.clear and size == 1 and out_path.exists():
        out_path.unlink()
        print(f"[INFO] Archivo {out_path} eliminado.")

    metrics = run_mpi(k=k, n_train=n_train, n_test=n_test)

    if rank == 0 and metrics is not None:
        extra = {
            "version": "mpi",
            "scaling": "weak",
            "frac": frac,
            "p": metrics["p"],
            "threads": 1,
            "workers": metrics["p"],
        }
        append_result_row(out_path, metrics, extra)

        workload = metrics["n_train"] * metrics["n_test"]
        time_per_pair = metrics["t_total"] / workload

        print(f"[MPI-WEAK] p={metrics['p']}, frac={frac:.2f}")
        print(
            f"n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
            f"k={metrics['k']}, acc={metrics['accuracy']:.4f}, "
            f"T_total={metrics['t_total']:.6f} s, "
            f"T_comp={metrics['t_compute']:.6f} s, "
            f"T_comm={metrics['t_comm']:.6f} s, "
            f"workload={workload}, time_per_pair={time_per_pair:.3e} s"
        )


if __name__ == "__main__":
    main()
