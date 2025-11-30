# experiments_mpi_weak.py
from mpi4py import MPI

from knn_mpi import run_mpi
from knn_sequential import load_data


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # p

    # 1. Obtener tamaño completo del dataset (solo root)
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

    # 2. Definir fracción según p (weak scaling simple)
    base_frac = 0.25
    frac = base_frac * size
    if frac > 1.0:
        frac = 1.0

    n_train = int(frac * n_train_full)
    n_test = int(frac * n_test_full)

    k = 3

    metrics = run_mpi(k=k, n_train=n_train, n_test=n_test)

    if rank == 0 and metrics is not None:
        workload = metrics["n_train"] * metrics["n_test"]
        time_per_pair = metrics["t_total"] / workload

        print(f"Resultados weak scaling (p = {size})")
        print("p,frac,n_train,n_test,k,acc,T_total,T_comp,T_comm,workload,time_per_pair")
        print(
            f"{size},"
            f"{frac:.2f},"
            f"{metrics['n_train']},"
            f"{metrics['n_test']},"
            f"{metrics['k']},"
            f"{metrics['accuracy']:.4f},"
            f"{metrics['t_total']:.6f},"
            f"{metrics['t_compute']:.6f},"
            f"{metrics['t_comm']:.6f},"
            f"{workload},"
            f"{time_per_pair:.6e}"
        )


if __name__ == "__main__":
    main()
