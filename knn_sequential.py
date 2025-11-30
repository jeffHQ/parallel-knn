# knn_sequential.py
from collections import Counter
import time

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# ---------- KNN BÁSICO (SECuencial) ----------

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distancia euclidiana entre dos vectores a y b.
    """
    diff = a - b
    return np.sqrt(np.dot(diff, diff))


def knn_predict_point(test_point: np.ndarray,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      k: int) -> int:
    """
    Predice la clase de un solo punto de prueba usando KNN secuencial.
    """
    # Distancias a todos los puntos de entrenamiento
    distances = [euclidean_distance(test_point, x) for x in X_train]

    # Índices de los k vecinos más cercanos
    k_indices = np.argsort(distances)[:k]

    # Etiquetas de esos vecinos
    k_labels = [y_train[i] for i in k_indices]

    # Voto mayoritario
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def knn_predict_batch(X_test: np.ndarray,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      k: int) -> np.ndarray:
    """
    Aplica KNN a todos los puntos de prueba (versión secuencial).
    """
    preds = [knn_predict_point(x, X_train, y_train, k) for x in X_test]
    return np.array(preds)


# ---------- CARGA DE DATOS Y EXPERIMENTO ----------

def load_data(test_size: float = 0.2,
              random_state: int = 42,
              n_train: int | None = None,
              n_test: int | None = None):
    """
    Carga el dataset digits y devuelve subconjuntos opcionales
    para experimentar con distintos tamaños.
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


def run_sequential(k: int = 3,
                   n_train: int | None = None,
                   n_test: int | None = None,
                   plot_examples: bool = False) -> dict:
    """
    Ejecuta el KNN secuencial, mide tiempo y devuelve métricas.
    """
    X_train, X_test, y_train, y_test = load_data(
        n_train=n_train,
        n_test=n_test,
    )

    n_tr, n_te = X_train.shape[0], X_test.shape[0]

    start_time = time.time()
    y_pred = knn_predict_batch(X_test, X_train, y_train, k)
    end_time = time.time()

    accuracy = float(np.mean(y_pred == y_test))
    t_total = end_time - start_time

    metrics = {
        "n_train": n_tr,
        "n_test": n_te,
        "k": k,
        "accuracy": accuracy,
        "t_total": t_total,
    }

    if plot_examples:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 5, figsize=(10, 4))
        for i, ax in enumerate(axes.flat):
            ax.imshow(X_test[i].reshape(8, 8), cmap="gray")
            ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
            ax.axis("off")
        plt.suptitle("Sample Predictions (Sequential KNN)")
        plt.tight_layout()
        plt.show()

    return metrics


# ---------- PUNTO DE ENTRADA ----------

if __name__ == "__main__":
    m = run_sequential(k=3, plot_examples=True)
    print(
        f"Sequential KNN | "
        f"n_train={m['n_train']}, n_test={m['n_test']}, "
        f"k={m['k']}, acc={m['accuracy']:.4f}, "
        f"time={m['t_total']:.4f} s"
    )
