from mpi4py import MPI
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from collections import Counter
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time() 

def euclid(a, b):
    return np.sqrt(np.sum((a - b)**2))

def knn_point(point, X_train, y_train, k):
    dist = [euclid(point, x) for x in X_train]
    idx = np.argsort(dist)[:k]
    lbls = [y_train[i] for i in idx]
    return Counter(lbls).most_common(1)[0][0]

if rank == 0:
    digits = load_digits()
    X, y = digits.data, digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    k = 1

    X_test_parts = np.array_split(X_test, size)

else:
    X_train = None
    y_train = None
    X_test_parts = None
    k = None
    y_test = None

# Enviar a todos
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
k = comm.bcast(k, root=0)
y_test = comm.bcast(y_test, root=0)

# Scatter
try:
    X_local = comm.scatter(X_test_parts, root=0)
except:
    X_local = [] 
# Predicciones locales
local_preds = []
for p in X_local:
    pred = knn_point(p, X_train, y_train, k)
    local_preds.append(pred)

# Recolectar predicciones
preds = comm.gather(local_preds, root=0)

# --- ROOT ---
if rank == 0:

    # Unir predicciones
    try:
        y_pred = np.concatenate(preds)
    except:
        print("Advertencia: fallback de predicciones.")
        y_pred = np.array([])

    end_time = time.time()

    if len(y_pred) == len(y_test):
        accuracy = np.mean(y_pred == y_test)
    else:
        accuracy = 0.0

    print(f"Número de procesos (size): {size}")
    print(f"Accuracy (Parallel): {accuracy:.4f}")
    print(f"Execution time (Parallel): {end_time - start_time:.4f} sec")

    # Ver si hay accuracy
    for i in range(10):
        print(f"Test {i}: pred:{y_pred[i] if len(y_pred)>i else 'NA'}  true:{y_test[i]}")
