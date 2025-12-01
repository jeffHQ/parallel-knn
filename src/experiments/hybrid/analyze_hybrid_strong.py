# src/experiments/hybrid/analyze_hybrid_strong.py

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


def load_hybrid_strong(path: Path):
    data = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["version"] != "hybrid":
                continue
            if row["scaling"] != "strong":
                continue
            data.append(
                {
                    "frac": float(row["frac"]),
                    "p": int(row["p"]),
                    "threads": int(row["threads"]),
                    "workers": int(row["workers"]),
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
    hyb_path = Path("results/hybrid/hybrid_strong.csv")

    if not seq_path.exists():
        print(f"[ERROR] No se encontró {seq_path}. Corre primero experiments_seq.py.")
        return
    if not hyb_path.exists():
        print(f"[ERROR] No se encontró {hyb_path}. Corre experiments_hybrid_strong.py con varias (p,threads).")
        return

    # 1. Secuencial: frac = 1.0
    seq_data = load_seq_results(seq_path)
    seq_full = next(d for d in seq_data if abs(d["frac"] - 1.0) < 1e-6)
    T_seq_full = seq_full["t_total"]
    flops_total = seq_full["flops"]

    print("Secuencial (frac=1.0):", seq_full)

    # 2. Híbrido: strong con frac = 1.0
    hyb_data = load_hybrid_strong(hyb_path)
    hyb_full = [d for d in hyb_data if abs(d["frac"] - 1.0) < 1e-6]

    if not hyb_full:
        print("[ERROR] No hay filas híbridas con frac=1.0 en hybrid_strong.csv.")
        return

    hyb_full = sorted(hyb_full, key=lambda d: d["workers"])

    workers = np.array([d["workers"] for d in hyb_full], dtype=int)
    ps = np.array([d["p"] for d in hyb_full], dtype=int)
    ts = np.array([d["threads"] for d in hyb_full], dtype=int)
    T_total = np.array([d["t_total"] for d in hyb_full])
    T_compute = np.array([d["t_compute"] for d in hyb_full])
    T_comm = np.array([d["t_comm"] for d in hyb_full])

    # 3. Speedup y eficiencia
    speedup = T_seq_full / T_total
    efficiency = speedup / workers

    print("\nTabla híbrido strong (frac=1.0):")
    print("p\tthreads\tW=p*threads\tT_total\tSpeedup\tEficiencia")
    for i in range(len(workers)):
        print(
            f"{ps[i]}\t{ts[i]}\t{workers[i]}\t"
            f"{T_total[i]:.6f}\t{speedup[i]:.3f}\t{efficiency[i]:.3f}"
        )

    # 4. FLOPs/s en región de cómputo
    flops_per_sec = flops_total / T_compute

    print("\nFLOPs/s híbrido (frac=1.0):")
    for i in range(len(workers)):
        print(
            f"(p={ps[i]}, threads={ts[i]}) -> FLOPs/s={flops_per_sec[i]:.3e}"
        )

    out_dir = Path("results/figures/hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)

 
    plt.figure()
    plt.plot(workers, speedup, marker="o")
    for i in range(len(workers)):
        plt.text(workers[i] * 1.01, speedup[i], f"({ps[i]},{ts[i]})", fontsize=8)
    plt.xlabel("Recursos totales (workers = p * threads)")
    plt.ylabel("Speedup")
    plt.title("Hybrid KNN (strong scaling): Speedup vs workers (frac=1.0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_speedup_vs_workers.png", dpi=300)
    plt.close()


    plt.figure()
    plt.plot(workers, efficiency, marker="o")
    for i in range(len(workers)):
        plt.text(workers[i] * 1.01, efficiency[i], f"({ps[i]},{ts[i]})", fontsize=8)
    plt.xlabel("Recursos totales (workers = p * threads)")
    plt.ylabel("Eficiencia")
    plt.title("Hybrid KNN (strong scaling): Eficiencia vs workers (frac=1.0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_efficiency_vs_workers.png", dpi=300)
    plt.close()

    # 3) Tiempos vs workers
    plt.figure()
    plt.plot(workers, T_total, marker="o", label="T_total")
    plt.plot(workers, T_compute, marker="o", label="T_compute")
    plt.plot(workers, T_comm, marker="o", label="T_comm")
    plt.xlabel("Recursos totales (workers = p * threads)")
    plt.ylabel("Tiempo (s)")
    plt.title("Hybrid KNN (strong scaling): Tiempos vs workers (frac=1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_times_vs_workers.png", dpi=300)
    plt.close()

    # 4) FLOPs/s vs workers
    plt.figure()
    plt.plot(workers, flops_per_sec, marker="o")
    for i in range(len(workers)):
        plt.text(workers[i] * 1.01, flops_per_sec[i], f"({ps[i]},{ts[i]})", fontsize=8)
    plt.xlabel("Recursos totales (workers = p * threads)")
    plt.ylabel("FLOPs/s")
    plt.title("Hybrid KNN (strong scaling): FLOPs/s vs workers (frac=1.0)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_flops_vs_workers.png", dpi=300)
    plt.close()

    print("\nSe generaron en results/figures/hybrid:")
    print("- hybrid_speedup_vs_workers.png")
    print("- hybrid_efficiency_vs_workers.png")
    print("- hybrid_times_vs_workers.png")
    print("- hybrid_flops_vs_workers.png")


if __name__ == "__main__":
    main()
