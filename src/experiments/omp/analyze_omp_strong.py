# src/experiments/omp/analyze_omp_strong.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_seq_results(path: Path):
    data = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(
                {
                    "version": row["version"],
                    "scaling": row["scaling"],
                    "frac": float(row["frac"]),
                    "n_train": int(row["n_train"]),
                    "n_test": int(row["n_test"]),
                    "k": int(row["k"]),
                    "accuracy": float(row["accuracy"]),
                    "t_total": float(row["t_total"]),
                    "flops": float(row["flops"]) if row["flops"] else 0.0,
                }
            )
    return data


def load_omp_strong(path: Path):
    data = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["version"] != "omp":
                continue
            if row["scaling"] != "strong":
                continue
            data.append(
                {
                    "frac": float(row["frac"]),
                    "threads": int(row["threads"]),
                    "n_train": int(row["n_train"]),
                    "n_test": int(row["n_test"]),
                    "k": int(row["k"]),
                    "accuracy": float(row["accuracy"]),
                    "t_total": float(row["t_total"]),
                    "t_compute": float(row["t_compute"]),
                    "t_comm": float(row["t_comm"]),
                }
            )
    return data


def main() -> None:
    seq_path = Path("results/seq/seq.csv")
    omp_path = Path("results/omp/omp_strong.csv")

    if not seq_path.exists():
        print(f"[ERROR] No se encontró {seq_path}. Corre primero experiments_seq.py.")
        return
    if not omp_path.exists():
        print(f"[ERROR] No se encontró {omp_path}. Corre primero experiments_omp_strong.py.")
        return

    # 1. Secuencial: fila con frac = 1.0
    seq_data = load_seq_results(seq_path)
    seq_full = next(d for d in seq_data if abs(d["frac"] - 1.0) < 1e-6)
    T_seq_full = seq_full["t_total"]
    flops_total = seq_full["flops"]

    print("Secuencial (frac=1.0):", seq_full)

    # 2. OMP: filas strong con frac = 1.0
    omp_data = load_omp_strong(omp_path)
    omp_full = [d for d in omp_data if abs(d["frac"] - 1.0) < 1e-6]

    if not omp_full:
        print("[ERROR] No hay filas OMP con frac=1.0 en omp_strong.csv.")
        return

    omp_full = sorted(omp_full, key=lambda d: d["threads"])

    threads = np.array([d["threads"] for d in omp_full], dtype=int)
    T_total_omp = np.array([d["t_total"] for d in omp_full])
    T_compute_omp = np.array([d["t_compute"] for d in omp_full])

    # 3. Speedup y eficiencia (respecto al secuencial)
    speedup = T_seq_full / T_total_omp
    efficiency = speedup / threads

    print("\nTabla OMP speedup/eficiencia (frac=1.0):")
    print("threads\tT_total\tSpeedup\tEficiencia")
    for i, t in enumerate(threads):
        print(
            f"{t}\t{T_total_omp[i]:.6f}\t{speedup[i]:.3f}\t{efficiency[i]:.3f}"
        )

    # 4. FLOPs/s usando el mismo flops_total que la versión secuencial
    flops_per_sec = flops_total / T_compute_omp

    print("\nFLOPs/s (OMP, región de cómputo, frac=1.0):")
    for i, t in enumerate(threads):
        print(f"threads={t}, FLOPs/s={flops_per_sec[i]:.3e}")

    out_dir = Path("results/figures/omp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Gráficos ---

    # 1) Speedup vs hilos
    plt.figure()
    plt.plot(threads, speedup, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Speedup")
    plt.title("OMP KNN (strong scaling): Speedup vs hilos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "omp_speedup_vs_threads.png", dpi=300)
    plt.close()

    # 2) Eficiencia vs hilos
    plt.figure()
    plt.plot(threads, efficiency, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Eficiencia")
    plt.title("OMP KNN (strong scaling): Eficiencia vs hilos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "omp_efficiency_vs_threads.png", dpi=300)
    plt.close()

    # 3) Tiempo total vs hilos
    plt.figure()
    plt.plot(threads, T_total_omp, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Tiempo total (s)")
    plt.title("OMP KNN (strong scaling): T_total vs hilos (frac=1.0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "omp_times_vs_threads.png", dpi=300)
    plt.close()

    # 4) FLOPs/s vs hilos
    plt.figure()
    plt.plot(threads, flops_per_sec, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("FLOPs/s")
    plt.title("OMP KNN (strong scaling): FLOPs/s vs hilos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "omp_flops_vs_threads.png", dpi=300)
    plt.close()

    print("\nSe generaron en results/figures/omp:")
    print("- omp_speedup_vs_threads.png")
    print("- omp_efficiency_vs_threads.png")
    print("- omp_times_vs_threads.png")
    print("- omp_flops_vs_threads.png")


if __name__ == "__main__":
    main()
