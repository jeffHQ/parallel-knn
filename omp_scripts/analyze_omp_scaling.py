# analyze_omp_scaling.py
import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def load_seq_results(path: str):
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


def load_omp_results(path: str):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Quitamos línea "Resultados OMP (threads = ...)"
    lines = [ln for ln in lines if not ln.startswith("Resultados OMP")]

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
            }
        )
    return data


def main():
    # 1. Secuencial (misma seq.csv que ya generaste antes)
    seq_data = load_seq_results("seq.csv")
    seq_full = next(d for d in seq_data if abs(d["frac"] - 1.0) < 1e-6)
    T_seq_full = seq_full["T_total"]
    n_train_full = seq_full["n_train"]
    n_test_full = seq_full["n_test"]

    print("Secuencial (frac=1.0):", seq_full)

    # 2. Cargar resultados OMP para distintos #hilos
    omp_files = {
        1: "omp_t1.csv",
        2: "omp_t2.csv",
        4: "omp_t4.csv",
        8: "omp_t8.csv",
    }

    omp_full = {}  # fila de frac=1.0 por cada #hilos

    for t, path in omp_files.items():
        data_t = load_omp_results(path)
        full_t = next(d for d in data_t if abs(d["frac"] - 1.0) < 1e-6)
        omp_full[t] = full_t

    threads = np.array(sorted(omp_full.keys()), dtype=int)
    T_total_omp = np.array([omp_full[t]["T_total"] for t in threads])

    # 3. Speedup y eficiencia (respecto al secuencial)
    speedup = T_seq_full / T_total_omp
    efficiency = speedup / threads

    print("\nTabla OMP speedup/eficiencia (frac=1.0):")
    print("threads\tT_total\tSpeedup\tEficiencia")
    for i, t in enumerate(threads):
        print(
            f"{t}\t{T_total_omp[i]:.6f}\t{speedup[i]:.3f}\t{efficiency[i]:.3f}"
        )

    # 4. FLOPs/s (región de cómputo; aquí asumimos T_total ≈ T_compute)
    digits = load_digits()
    d = digits.data.shape[1]  # 64
    flops_per_distance = 3 * d + 1
    flops_total = n_train_full * n_test_full * flops_per_distance

    flops_per_sec = flops_total / T_total_omp

    print("\nFLOPs/s (OMP, frac=1.0):")
    for i, t in enumerate(threads):
        print(f"threads={t}, FLOPs/s={flops_per_sec[i]:.3e}")

    # 5. Gráficos

    # 5.1 Speedup vs hilos
    plt.figure()
    plt.plot(threads, speedup, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Speedup")
    plt.title("Speedup vs número de hilos (KNN OMP)")
    plt.grid(True)
    plt.savefig("omp_speedup_vs_threads.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.2 Eficiencia vs hilos
    plt.figure()
    plt.plot(threads, efficiency, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Eficiencia")
    plt.title("Eficiencia vs número de hilos (KNN OMP)")
    plt.grid(True)
    plt.savefig("omp_efficiency_vs_threads.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.3 Tiempo total vs hilos
    plt.figure()
    plt.plot(threads, T_total_omp, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Tiempo total (s)")
    plt.title("Tiempo total vs número de hilos (KNN OMP, frac=1.0)")
    plt.grid(True)
    plt.savefig("omp_times_vs_threads.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.4 FLOPs/s vs hilos
    plt.figure()
    plt.plot(threads, flops_per_sec, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("FLOPs/s")
    plt.title("FLOPs/s vs número de hilos (KNN OMP)")
    plt.grid(True)
    plt.savefig("omp_flops_vs_threads.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSe generaron:")
    print("- omp_speedup_vs_threads.png")
    print("- omp_efficiency_vs_threads.png")
    print("- omp_times_vs_threads.png")
    print("- omp_flops_vs_threads.png")


if __name__ == "__main__":
    main()
