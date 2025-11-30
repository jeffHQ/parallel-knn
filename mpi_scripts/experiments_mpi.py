# experiments_mpi.py
from mpi4py import MPI

from knn_mpi import run_mpi
from knn_sequential import load_data


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # --- 1. Obtener tamaño máximo del dataset (solo root usa load_data) ---
    if rank == 0:
        X_train_full, X_test_full, _, _ = load_data(
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

    # --- 2. Definir fracciones de tamaño de problema ---
    fractions = [0.25, 0.5, 0.75, 1.0]
    k = 3

    results = []

    for frac in fractions:
        n_train = int(frac * n_train_full)
        n_test = int(frac * n_test_full)

        metrics = run_mpi(k=k, n_train=n_train, n_test=n_test)

        if rank == 0 and metrics is not None:
            metrics["frac"] = frac
            results.append(metrics)

    # --- 3. Imprimir tabla de resultados (solo root) ---
    if rank == 0:
        print(f"Resultados MPI (p = {size})")
        print("frac,n_train,n_test,k,acc,T_total,T_comp,T_comm")
        for m in results:
            print(
                f"{m['frac']:.2f},"
                f"{m['n_train']},"
                f"{m['n_test']},"
                f"{m['k']},"
                f"{m['accuracy']:.4f},"
                f"{m['t_total']:.6f},"
                f"{m['t_compute']:.6f},"
                f"{m['t_comm']:.6f}"
            )


if __name__ == "__main__":
    main()
