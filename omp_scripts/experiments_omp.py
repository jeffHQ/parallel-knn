# experiments_omp.py
import argparse

from knn_omp import run_omp
from knn_sequential import load_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=1,
                        help="Número de hilos para la versión OMP (joblib)")
    args = parser.parse_args()
    n_threads = args.threads

    # Obtener tamaño completo del dataset
    X_train_full, X_test_full, _, _ = load_data(
        n_train=None,
        n_test=None,
    )
    n_train_full = X_train_full.shape[0]
    n_test_full = X_test_full.shape[0]

    fractions = [0.25, 0.5, 0.75, 1.0]
    k = 3

    print(f"Resultados OMP (threads = {n_threads})")
    print("frac,n_train,n_test,k,acc,T_total")

    for frac in fractions:
        n_train = int(frac * n_train_full)
        n_test = int(frac * n_test_full)

        metrics = run_omp(
            k=k,
            n_train=n_train,
            n_test=n_test,
            n_threads=n_threads,
        )

        print(
            f"{frac:.2f},"
            f"{metrics['n_train']},"
            f"{metrics['n_test']},"
            f"{metrics['k']},"
            f"{metrics['accuracy']:.4f},"
            f"{metrics['t_total']:.6f}"
        )


if __name__ == "__main__":
    main()
