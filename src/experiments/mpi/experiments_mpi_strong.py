# src/experiments/mpi/experiments_mpi_strong.py

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
        description="MPI Strong Scaling: problema fijo, variar procesos"
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

    fractions = args.fractions
    k = args.k

    out_path = Path("results/mpi/mpi_strong.csv")

    # Solo en p=1 con --clear borramos el archivo para reiniciar el experimento completo
    if rank == 0 and args.clear and size == 1 and out_path.exists():
        out_path.unlink()
        print(f"[INFO] Archivo {out_path} eliminado.")

    for frac in fractions:
        n_train = int(frac * n_train_full)
        n_test = int(frac * n_test_full)

        metrics = run_mpi(k=k, n_train=n_train, n_test=n_test)

        if rank == 0 and metrics is not None:
            extra = {
                "version": "mpi",
                "scaling": "strong",
                "frac": frac,
                "p": metrics["p"],      # debería ser igual a size
                "threads": 1,
                "workers": metrics["p"],  # solo procesos (sin threads)
            }
            append_result_row(out_path, metrics, extra)

            print(
                f"[MPI-STRONG] p={metrics['p']}, frac={frac:.2f}, "
                f"n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
                f"acc={metrics['accuracy']:.4f}, T_total={metrics['t_total']:.6f} s"
            )


if __name__ == "__main__":
    main()
