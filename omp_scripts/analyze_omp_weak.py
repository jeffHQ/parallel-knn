# analyze_omp_weak.py
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_omp_weak(path: str):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    data = []
    for r in rows:
        data.append(
            {
                "threads": int(r["threads"]),
                "frac": float(r["frac"]),
                "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]),
                "k": int(r["k"]),
                "acc": float(r["acc"]),
                "T_total": float(r["T_total"]),
                "workload": int(r["workload"]),
                "time_per_pair": float(r["time_per_pair"]),
            }
        )
    return sorted(data, key=lambda x: x["threads"])


def main():
    data = load_omp_weak("omp_weak.csv")

    threads = np.array([d["threads"] for d in data])
    fracs = np.array([d["frac"] for d in data])
    T_total = np.array([d["T_total"] for d in data])
    workload = np.array([d["workload"] for d in data])
    time_per_pair = np.array([d["time_per_pair"] for d in data])

    print("Tabla OMP weak scaling:")
    print("threads\tfrac\tworkload\tT_total\ttime_per_pair")
    for d in data:
        print(
            f"{d['threads']}\t{d['frac']:.2f}\t{d['workload']}\t"
            f"{d['T_total']:.6f}\t{d['time_per_pair']:.3e}"
        )

    # Gráfico 1: T_total vs threads
    plt.figure()
    plt.plot(threads, T_total, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Tiempo total (s)")
    plt.title("Weak scaling OMP: T_total vs número de hilos")
    plt.grid(True)
    plt.savefig("omp_weak_Ttotal_vs_threads.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico 2: tiempo normalizado vs threads
    plt.figure()
    plt.plot(threads, time_per_pair, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Tiempo por par train-test (s)")
    plt.title("Weak scaling OMP: tiempo normalizado vs hilos")
    plt.grid(True)
    plt.savefig("omp_weak_time_per_pair_vs_threads.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSe generaron:")
    print("- omp_weak_Ttotal_vs_threads.png")
    print("- omp_weak_time_per_pair_vs_threads.png")


if __name__ == "__main__":
    main()
