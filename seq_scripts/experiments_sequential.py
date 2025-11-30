# experiments_sequential.py
from omp_scripts.knn_sequential import run_sequential, load_data


def main():
    # Obtenemos tamaño completo del dataset
    X_train_full, X_test_full, _, _ = load_data(
        n_train=None,
        n_test=None,
    )
    n_train_full = X_train_full.shape[0]
    n_test_full = X_test_full.shape[0]

    fractions = [0.25, 0.5, 0.75, 1.0]
    k = 3

    print("frac,n_train,n_test,k,acc,T_total")

    for frac in fractions:
        n_train = int(frac * n_train_full)
        n_test = int(frac * n_test_full)

        metrics = run_sequential(
            k=k,
            n_train=n_train,
            n_test=n_test,
            plot_examples=False,
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
