# src/experiments/omp/experiments_omp_strong.py

from __future__ import annotations

import argparse
from pathlib import Path

from core.data_utils import load_digits_data
from core.csv_utils import append_result_row
from methods.knn_omp import run_omp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMP Strong Scaling: problema fijo, variar threads"
    )
    parser.add_argument(
        "--threads-list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Lista de threads a probar (default: 1 2 4 8 16)",
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
        help="Limpiar archivo de resultados antes de empezar",
    )
    args = parser.parse_args()

    # 1. Tamaño completo del dataset
    X_train_full, X_test_full, _, _ = load_digits_data(
        n_train=None,
        n_test=None,
    )
    n_train_full = X_train_full.shape[0]
    n_test_full = X_test_full.shape[0]

    # 2. Configuración de experimento
    threads_list = args.threads_list
    fractions = args.fractions
    k = args.k

    out_path = Path("results/omp/omp_strong.csv")

    # Limpiamos archivo anterior si se especifica
    if args.clear and out_path.exists():
        out_path.unlink()
        print(f"[INFO] Archivo {out_path} eliminado.")

    # 3. Bucle sobre #hilos y fracciones
    for n_threads in threads_list:
        for frac in fractions:
            n_train = int(frac * n_train_full)
            n_test = int(frac * n_test_full)

            metrics = run_omp(
                k=k,
                n_train=n_train,
                n_test=n_test,
                n_threads=n_threads,
            )

            extra = {
                "version": "omp",
                "scaling": "strong",
                "frac": frac,
                "p": 1,                      # solo un proceso
                "threads": n_threads,
                "workers": n_threads,        # workers = 1 * threads
            }

            append_result_row(out_path, metrics, extra)

            print(
                f"[OMP-STRONG] threads={n_threads}, frac={frac:.2f}, "
                f"n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
                f"acc={metrics['accuracy']:.4f}, T_total={metrics['t_total']:.6f} s"
            )


if __name__ == "__main__":
    main()
