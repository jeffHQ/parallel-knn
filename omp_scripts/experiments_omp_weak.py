# experiments_omp_weak.py
from knn_omp import run_omp
from knn_sequential import load_data


def main():
    # Tamaño completo del dataset
    X_train_full, X_test_full, _, _ = load_data(
        n_train=None,
        n_test=None,
    )
    n_train_full = X_train_full.shape[0]
    n_test_full = X_test_full.shape[0]

    base_frac = 0.25
    threads_list = [1, 2, 4, 8]
    k = 3

    print("threads,frac,n_train,n_test,k,acc,T_total,workload,time_per_pair")

    for t in threads_list:
        frac = base_frac * t
        if frac > 1.0:
            frac = 1.0

        n_train = int(frac * n_train_full)
        n_test = int(frac * n_test_full)

        metrics = run_omp(
            k=k,
            n_train=n_train,
            n_test=n_test,
            n_threads=t,
        )

        workload = metrics["n_train"] * metrics["n_test"]
        time_per_pair = metrics["t_total"] / workload

        print(
            f"{t},"
            f"{frac:.2f},"
            f"{metrics['n_train']},"
            f"{metrics['n_test']},"
            f"{metrics['k']},"
            f"{metrics['accuracy']:.4f},"
            f"{metrics['t_total']:.6f},"
            f"{workload},"
            f"{time_per_pair:.6e}"
        )


if __name__ == "__main__":
    main()
