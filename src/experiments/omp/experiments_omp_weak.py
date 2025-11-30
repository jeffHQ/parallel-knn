# src/experiments/omp/experiments_omp_weak.py

from __future__ import annotations

import argparse
from pathlib import Path

from core.data_utils import load_digits_data
from core.csv_utils import append_result_row
from methods.knn_omp import run_omp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OMP Weak Scaling: escalar problema con threads"
    )
    parser.add_argument(
        "--threads-list",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Lista de threads a probar (default: 1 2 4 8 16)",
    )
    parser.add_argument(
        "--base-frac",
        type=float,
        default=0.25,
        help="Fracción base para W=1 (default: 0.25)",
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

    base_frac = args.base_frac
    threads_list = args.threads_list
    k = args.k

    out_path = Path("results/omp/omp_weak.csv")

    # Limpiamos archivo previo si se especifica
    if args.clear and out_path.exists():
        out_path.unlink()
        print(f"[INFO] Archivo {out_path} eliminado.")

    for n_threads in threads_list:
        frac = base_frac * n_threads
        if frac > 1.0:
            frac = 1.0

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
            "scaling": "weak",
            "frac": frac,
            "p": 1,
            "threads": n_threads,
            "workers": n_threads,
        }

        append_result_row(out_path, metrics, extra)

        workload = metrics["n_train"] * metrics["n_test"]
        time_per_pair = metrics["t_total"] / workload

        print(
            f"[OMP-WEAK] threads={n_threads}, frac={frac:.2f}, "
            f"n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
            f"k={metrics['k']}, acc={metrics['accuracy']:.4f}, "
            f"T_total={metrics['t_total']:.6f} s, "
            f"workload={workload}, time_per_pair={time_per_pair:.3e} s"
        )


if __name__ == "__main__":
    main()
