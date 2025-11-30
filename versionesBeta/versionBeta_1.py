import numpy as np
import time
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

# Inicialización de MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Funciones de KNN ---
def euclidean_distance(a, b):
    # Calcula la distancia euclidiana entre dos puntos
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(test_point, X_train, y_train, k):
    # Calcula distancias a todos los puntos "x" de entrenamiento "x_train"
    distances = [euclidean_distance(test_point, x) for x in X_train]
    
    # Obtiene los índices de los k puntos más cercanos
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    
    # Encuentra la etiqueta más común
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]

# --- Proceso Raíz (rank 0) ---
if rank == 0:
    start_time = time.time()
    
    # 1. Cargar y dividir los datos (Parámetros Iniciales)
    digits = load_digits()
    X_data, y_data = digits.data, digits.target
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    k = 3
    
    # Determinar el tamaño de la porción de X_test para cada proceso
    # np.array_split divide el arreglo en 'size' subarreglos de tamaño casi igual
    X_test_splits = np.array_split(X_test, size)
    
    # Se inicializa X_test_local (solo para evitar errores de referencia en el broadcast/scatter)
    X_test_local = None 
    
else:
    # Inicializa variables para otros procesos
    X_train, y_train, X_test_local, k = None, None, None, None
    X_test_splits = None

# 2. Difusión (comm.bcast) de parámetros de entrenamiento y k
# Estos datos son necesarios para que cada proceso haga la predicción en su subconjunto
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)

# 3. Distribución (comm.scatter) del conjunto de prueba
# Cada proceso recibe su porción de X_test
X_test_local = comm.scatter(X_test_splits, root=0)

# 4. Cálculo Local Distribuido (dist0...distp-1 sort k-nearest)
# Cada proceso calcula predicciones para su porción de X_test
y_pred_local = [knn_predict(x, X_train, y_train, k) for x in X_test_local]
y_pred_local = np.array(y_pred_local) # Convertir a numpy array para Gather

# 5. Recolección (comm.gather) de predicciones
# El proceso raíz reúne todas las predicciones
y_pred_all = comm.gather(y_pred_local, root=0)

# --- Proceso Raíz (rank 0) ---
if rank == 0:
    # 6. Aplicar Mayoría (Aplicación de Mayoría) - Implícito en la recolección
    # Concatenar las predicciones locales en el orden correcto
    y_pred = np.concatenate(y_pred_all)
    
    end_time = time.time()
    
    # Evaluar
    accuracy = np.mean(y_pred == y_test)
    
    print(f"Número de procesos (size): {size}")
    print(f"Accuracy (Parallel): {accuracy:.4f}")
    print(f"Execution time (Parallel): {end_time - start_time:.4f} sec")

    # Visualización
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i].reshape(8, 8), cmap='gray')
        ax.set_title(f"Pred: {y_pred[i]}\nTrue: {y_test[i]}")
        ax.axis('off')
    plt.suptitle("Sample Predictions (Parallel KNN)")
    plt.tight_layout()
    plt.show()