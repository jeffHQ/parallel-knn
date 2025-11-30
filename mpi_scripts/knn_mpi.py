# knn_mpi.py
from collections import Counter

import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return np.sqrt(np.dot(diff, diff))


def load_data_root(n_train: int | None = None,
                   n_test: int | None = None,
                   test_size: float = 0.2,
                   random_state: int = 42):
    """
    Solo la ejecuta el rank 0.
    Carga digits, hace train/test split y recorta tamaños si se pide.
    """
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=test_size,
        random_state=random_state,
        stratify=digits.target,
    )

    if n_train is not None:
        X_train = X_train[:n_train]
        y_train = y_train[:n_train]

    if n_test is not None:
        X_test = X_test[:n_test]
        y_test = y_test[:n_test]

    return X_train, X_test, y_train, y_test


def run_mpi(k: int = 3,
            n_train: int | None = None,
            n_test: int | None = None) -> dict | None:
    """
    Versión paralela de KNN usando MPI.
    Reparte el conjunto de entrenamiento entre procesos (scatter) y
    replica el conjunto de prueba (bcast).

    Devuelve métricas SOLO en el rank 0. En los demás ranks devuelve None.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # ---------- 1. Carga de datos y particionado (solo root) ----------

    if rank == 0:
        X_train_full, X_test_full, y_train_full, y_test_full = load_data_root(
            n_train=n_train,
            n_test=n_test,
        )
        n_train_global = X_train_full.shape[0]
        n_test_global = X_test_full.shape[0]

        X_train_chunks = np.array_split(X_train_full, size)
        y_train_chunks = np.array_split(y_train_full, size)

        meta = (n_train_global, n_test_global)
    else:
        X_train_chunks = None
        y_train_chunks = None
        X_test_full = None
        y_test_full = None
        meta = None

    # Todos saben cuántos train/test globales hay
    n_train_global, n_test_global = comm.bcast(meta, root=0)

    # ---------- 2. Medir tiempo de comunicación inicial (scatter + bcast) ----------

    t_comm_local = 0.0
    t_comp_local = 0.0

    t0_comm = MPI.Wtime()
    # Cada proceso recibe su parte del train
    X_train_local = comm.scatter(X_train_chunks, root=0)
    y_train_local = comm.scatter(y_train_chunks, root=0)
    # Todos reciben copia del test y sus etiquetas
    X_test_local = comm.bcast(X_test_full, root=0)
    y_test_local = comm.bcast(y_test_full, root=0)
    t1_comm = MPI.Wtime()
    t_comm_local += (t1_comm - t0_comm)

    # ---------- 3. Bucle principal de clasificación ----------

    comm.Barrier()
    t_start = MPI.Wtime()

    predictions = []  # solo se llenará en el rank 0

    for j in range(n_test_global):
        test_point = X_test_local[j]

        # --- Cómputo local: distancias a mi chunk de train ---
        t0_comp = MPI.Wtime()
        dists = [euclidean_distance(test_point, x) for x in X_train_local]
        k_local = min(k, len(dists))
        local_indices = np.argsort(dists)[:k_local]
        local_neighbors = [(dists[i], y_train_local[i]) for i in local_indices]
        t1_comp = MPI.Wtime()
        t_comp_local += (t1_comp - t0_comp)

        # --- Comunicación: gather de vecinos hacia root ---
        t0_comm = MPI.Wtime()
        all_neighbors = comm.gather(local_neighbors, root=0)
        t1_comm = MPI.Wtime()
        t_comm_local += (t1_comm - t0_comm)

        # --- Fusión y voto mayoritario (solo root) ---
        if rank == 0:
            merged = [item for sublist in all_neighbors for item in sublist]
            merged.sort(key=lambda x: x[0])  # ordenar por distancia
            top_k = merged[:k]
            labels = [lab for (_, lab) in top_k]
            pred_label = Counter(labels).most_common(1)[0][0]
            predictions.append(pred_label)

    comm.Barrier()
    t_end = MPI.Wtime()

    # ---------- 4. Reducir tiempos (máximo entre ranks) ----------

    t_total_local = t_end - t_start

    t_total = comm.reduce(t_total_local, op=MPI.MAX, root=0)
    t_comp = comm.reduce(t_comp_local, op=MPI.MAX, root=0)
    t_comm = comm.reduce(t_comm_local, op=MPI.MAX, root=0)

    # ---------- 5. Métricas (solo root) ----------

    if rank == 0:
        y_pred = np.array(predictions)
        accuracy = float(np.mean(y_pred == y_test_local))

        metrics = {
            "p": size,
            "n_train": n_train_global,
            "n_test": n_test_global,
            "k": k,
            "accuracy": accuracy,
            "t_total": t_total,
            "t_compute": t_comp,
            "t_comm": t_comm,
        }
        return metrics

    return None


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    metrics = run_mpi(k=3)

    if rank == 0 and metrics is not None:
        print(
            f"MPI KNN | p={metrics['p']}, "
            f"n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
            f"k={metrics['k']}, acc={metrics['accuracy']:.4f}, "
            f"T_total={metrics['t_total']:.4f} s, "
            f"T_comp={metrics['t_compute']:.4f} s, "
            f"T_comm={metrics['t_comm']:.4f} s"
        )
