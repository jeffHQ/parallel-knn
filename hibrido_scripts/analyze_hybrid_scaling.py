# analyze_hybrid_scaling.py
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


def load_hybrid_results(path: str):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Quitamos la línea "Resultados Hybrid (p = ..., threads = ...)"
    lines = [ln for ln in lines if not ln.startswith("Resultados Hybrid")]

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
                "p": int(row["p"]),
                "threads": int(row["threads"]),
            }
        )
    return data


def main():
    # 1. Secuencial full
    seq_data = load_seq_results("seq.csv")
    seq_full = next(d for d in seq_data if abs(d["frac"] - 1.0) < 1e-6)
    T_seq_full = seq_full["T_total"]
    n_train_full = seq_full["n_train"]
    n_test_full = seq_full["n_test"]

    print("Secuencial (frac=1.0):", seq_full)

    # 2. Archivos híbridos (usa los nombres que generaste arriba)
    hybrid_files = {
        (1, 2): "hybrid_p1_t2.csv",
        (2, 2): "hybrid_p2_t2.csv",
        (4, 2): "hybrid_p4_t2.csv",
    }

    configs = []  # lista de dicts con info de cada (p,t)

    for (p, t), path in hybrid_files.items():
        data_pt = load_hybrid_results(path)
        full_pt = next(d for d in data_pt if abs(d["frac"] - 1.0) < 1e-6)
        configs.append(full_pt)

    # Ordenar por recursos totales W = p * threads
    configs.sort(key=lambda d: d["p"] * d["threads"])

    Ws = np.array([c["p"] * c["threads"] for c in configs])
    T_total = np.array([c["T_total"] for c in configs])
    T_comp = np.array([c["T_comp"] for c in configs])
    T_comm = np.array([c["T_comm"] for c in configs])
    ps = np.array([c["p"] for c in configs])
    ts = np.array([c["threads"] for c in configs])

    # 3. Speedup y eficiencia
    speedup = T_seq_full / T_total
    efficiency = speedup / Ws

    print("\nTabla híbrido (frac=1.0):")
    print("p\tthreads\tW=p*t\tT_total\tSpeedup\tEficiencia")
    for i in range(len(configs)):
        print(
            f"{ps[i]}\t{ts[i]}\t{Ws[i]}\t"
            f"{T_total[i]:.6f}\t{speedup[i]:.3f}\t{efficiency[i]:.3f}"
        )

    # 4. FLOPs/s (usando T_comp de la región paralela)
    digits = load_digits()
    d = digits.data.shape[1]  # 64
    flops_per_distance = 3 * d + 1
    flops_total = n_train_full * n_test_full * flops_per_distance
    flops_per_sec = flops_total / T_comp

    print("\nFLOPs/s híbrido (frac=1.0):")
    for i in range(len(configs)):
        print(
            f"(p={ps[i]}, threads={ts[i]}) -> "
            f"FLOPs/s={flops_per_sec[i]:.3e}"
        )

    # 5. Gráficos

    # 5.1 Speedup vs W
    plt.figure()
    plt.plot(Ws, speedup, marker="o")
    for i in range(len(Ws)):
        plt.text(Ws[i] * 1.01, speedup[i],
                 f"({ps[i]},{ts[i]})", fontsize=8)
    plt.xlabel("Recursos totales W = p * threads")
    plt.ylabel("Speedup")
    plt.title("Hybrid: Speedup vs recursos totales (frac=1.0)")
    plt.grid(True)
    plt.savefig("hybrid_speedup_vs_W.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.2 Eficiencia vs W
    plt.figure()
    plt.plot(Ws, efficiency, marker="o")
    for i in range(len(Ws)):
        plt.text(Ws[i] * 1.01, efficiency[i],
                 f"({ps[i]},{ts[i]})", fontsize=8)
    plt.xlabel("Recursos totales W = p * threads")
    plt.ylabel("Eficiencia")
    plt.title("Hybrid: Eficiencia vs recursos totales (frac=1.0)")
    plt.grid(True)
    plt.savefig("hybrid_efficiency_vs_W.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.3 Tiempos vs W
    plt.figure()
    plt.plot(Ws, T_total, marker="o", label="T_total")
    plt.plot(Ws, T_comp, marker="o", label="T_compute")
    plt.plot(Ws, T_comm, marker="o", label="T_comm")
    plt.xlabel("Recursos totales W = p * threads")
    plt.ylabel("Tiempo (s)")
    plt.title("Hybrid: tiempos vs recursos totales (frac=1.0)")
    plt.grid(True)
    plt.legend()
    plt.savefig("hybrid_times_vs_W.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5.4 FLOPs/s vs W
    plt.figure()
    plt.plot(Ws, flops_per_sec, marker="o")
    for i in range(len(Ws)):
        plt.text(Ws[i] * 1.01, flops_per_sec[i],
                 f"({ps[i]},{ts[i]})", fontsize=8)
    plt.xlabel("Recursos totales W = p * threads")
    plt.ylabel("FLOPs/s")
    plt.title("Hybrid: FLOPs/s vs recursos totales (frac=1.0)")
    plt.grid(True)
    plt.savefig("hybrid_flops_vs_W.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSe generaron:")
    print("- hybrid_speedup_vs_W.png")
    print("- hybrid_efficiency_vs_W.png")
    print("- hybrid_times_vs_W.png")
    print("- hybrid_flops_vs_W.png")


if __name__ == "__main__":
    main()
