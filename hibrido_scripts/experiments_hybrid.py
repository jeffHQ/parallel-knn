# experiments_hybrid.py
import argparse

from mpi4py import MPI

from knn_hybrid import run_hybrid
from knn_sequential import load_data


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()  # número de procesos (p)

    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=2,
                        help="Número de hilos por proceso en el híbrido")
    args = parser.parse_args()
    n_threads = args.threads

    # 1. Tamaño completo del dataset (solo root)
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

    fractions = [0.25, 0.5, 0.75, 1.0]
    k = 3

    results = []

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
            metrics["frac"] = frac
            results.append(metrics)

    # 2. Imprimir CSV (solo root)
    if rank == 0:
        print(f"Resultados Hybrid (p = {size}, threads = {n_threads})")
        print("frac,n_train,n_test,k,acc,"
              "T_total,T_comp,T_comm,p,threads")

        for m in results:
            print(
                f"{m['frac']:.2f},"
                f"{m['n_train']},"
                f"{m['n_test']},"
                f"{m['k']},"
                f"{m['accuracy']:.4f},"
                f"{m['t_total']:.6f},"
                f"{m['t_compute']:.6f},"
                f"{m['t_comm']:.6f},"
                f"{m['p']},"
                f"{m['threads']}"
            )


if __name__ == "__main__":
    main()
