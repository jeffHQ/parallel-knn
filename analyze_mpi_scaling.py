# analyze_mpi_scaling.py
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def load_seq_results(path: str):
    """
    Lee seq.csv generado por experiments_sequential.py
    """
    data = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "frac": float(row["frac"]),
                    "n_train": int(row["n_train"]),
                    "n_test": int(row["n_test"]),
                    "k": int(row["k"]),
                    "acc": float(row["acc"]),
                    "T_total": float(row["T_total"]),
                }
            )
    return data


def load_mpi_results(path: str):
    """
    Lee mpi_pX.csv generado por experiments_mpi.py
    Maneja el encabezado 'Resultados MPI (p = ...)'.
    """
    with open(path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Quitamos la línea tipo "Resultados MPI (p = X)" si existe
    lines = [ln for ln in lines if not ln.startswith("Resultados MPI")]

    # Usamos DictReader sobre las líneas restantes
    reader = csv.DictReader(lines)
    data = []
    for row in reader:
        data.append(
            {
                "frac": float(row["frac"]),
                "n_train": int(row["n_train"]),
                "n_test": int(row["n_test"]),
                "k": int(row["k"]),
                "acc": float(row["acc"]),
                "T_total": float(row["T_total"]),
                "T_comp": float(row["T_comp"]),
                "T_comm": float(row["T_comm"]),
            }
        )
    return data


def main():
    # ---------- 1. Cargar resultados secuenciales ----------
    seq_data = load_seq_results("seq.csv")

    # Tomamos como referencia la fila con frac = 1.0 (tamaño completo)
    seq_full = next(d for d in seq_data if abs(d["frac"] - 1.0) < 1e-6)
    T_seq_full = seq_full["T_total"]
    n_train_full = seq_full["n_train"]
    n_test_full = seq_full["n_test"]

    print("Secuencial (frac=1.0):", seq_full)

    # ---------- 2. Cargar resultados MPI para varios p ----------
    mpi_files = {
        1: "mpi_p1.csv",
        2: "mpi_p2.csv",
        4: "mpi_p4.csv",
        8: "mpi_p8.csv",
    }

    mpi_full = {}  # por cada p guardamos la fila con frac=1.0

    for p, path in mpi_files.items():
        data_p = load_mpi_results(path)
        # Fila de tamaño completo (frac = 1.0)
        full_p = next(d for d in data_p if abs(d["frac"] - 1.0) < 1e-6)
        mpi_full[p] = full_p

    # ---------- 3. Construir vectores para cálculo ----------
    ps = np.array(sorted(mpi_full.keys()), dtype=int)
    T_total_mpi = np.array([mpi_full[p]["T_total"] for p in ps])
    T_comp_mpi = np.array([mpi_full[p]["T_comp"] for p in ps])
    T_comm_mpi = np.array([mpi_full[p]["T_comm"] for p in ps])

    # ---------- 4. Speedup y eficiencia ----------
    speedup = T_seq_full / T_total_mpi
    efficiency = speedup / ps

    print("\nTabla speedup/eficiencia (frac=1.0):")
    print("p\tT_total\tSpeedup\tEficiencia")
    for i, p in enumerate(ps):
        print(
            f"{p}\t{T_total_mpi[i]:.6f}\t{speedup[i]:.3f}\t{efficiency[i]:.3f}"
        )

    # ---------- 5. FLOPs/s de la región paralela ----------
    digits = load_digits()
    d = digits.data.shape[1]  # número de características (64)
    flops_per_distance = 3 * d + 1

    flops_total = n_train_full * n_test_full * flops_per_distance
    flops_per_sec = flops_total / T_comp_mpi

    print("\nFLOPs/s (región de cómputo de distancias, frac=1.0):")
    for i, p in enumerate(ps):
        print(f"p={p}, FLOPs/s={flops_per_sec[i]:.3e}")

    # ---------- 6. Gráficos ----------

    # 6.1 Speedup vs p
    plt.figure()
    plt.plot(ps, speedup, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Speedup")
    plt.title("Speedup vs número de procesos (KNN MPI)")
    plt.grid(True)
    plt.savefig("speedup_vs_p.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6.2 Eficiencia vs p
    plt.figure()
    plt.plot(ps, efficiency, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Eficiencia")
    plt.title("Eficiencia vs número de procesos (KNN MPI)")
    plt.grid(True)
    plt.savefig("efficiency_vs_p.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6.3 Tiempos vs p
    plt.figure()
    plt.plot(ps, T_total_mpi, marker="o", label="T_total")
    plt.plot(ps, T_comp_mpi, marker="o", label="T_compute")
    plt.plot(ps, T_comm_mpi, marker="o", label="T_comm")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Tiempo (s)")
    plt.title("Tiempos vs número de procesos (KNN MPI, frac=1.0)")
    plt.grid(True)
    plt.legend()
    plt.savefig("times_vs_p.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 6.4 FLOPs/s vs p
    plt.figure()
    plt.plot(ps, flops_per_sec, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("FLOPs/s")
    plt.title("FLOPs/s vs número de procesos (región de cómputo)")
    plt.grid(True)
    plt.savefig("flops_vs_p.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSe generaron los archivos:")
    print("- speedup_vs_p.png")
    print("- efficiency_vs_p.png")
    print("- times_vs_p.png")
    print("- flops_vs_p.png")


if __name__ == "__main__":
    main()
