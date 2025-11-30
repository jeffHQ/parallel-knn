# src/experiments/mpi/analyze_mpi_strong.py

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


def load_mpi_strong(path: Path):
    data = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["version"] != "mpi":
                continue
            if row["scaling"] != "strong":
                continue
            data.append(
                {
                    "frac": float(row["frac"]),
                    "p": int(row["p"]),
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
    mpi_path = Path("results/mpi/mpi_strong.csv")

    if not seq_path.exists():
        print(f"[ERROR] No se encontró {seq_path}. Corre primero experiments_seq.py.")
        return
    if not mpi_path.exists():
        print(f"[ERROR] No se encontró {mpi_path}. Corre primero experiments_mpi_strong.py con varios p.")
        return

    # 1. Secuencial: fila con frac = 1.0
    seq_data = load_seq_results(seq_path)
    seq_full = next(d for d in seq_data if abs(d["frac"] - 1.0) < 1e-6)
    T_seq_full = seq_full["t_total"]
    flops_total = seq_full["flops"]
    n_train_full = seq_full["n_train"]
    n_test_full = seq_full["n_test"]

    print("Secuencial (frac=1.0):", seq_full)

    # 2. MPI: filas strong con frac = 1.0
    mpi_data = load_mpi_strong(mpi_path)
    mpi_full = [d for d in mpi_data if abs(d["frac"] - 1.0) < 1e-6]

    if not mpi_full:
        print("[ERROR] No hay filas MPI con frac=1.0 en mpi_strong.csv.")
        return

    mpi_full = sorted(mpi_full, key=lambda d: d["p"])

    ps = np.array([d["p"] for d in mpi_full], dtype=int)
    T_total_mpi = np.array([d["t_total"] for d in mpi_full])
    T_comp_mpi = np.array([d["t_compute"] for d in mpi_full])
    T_comm_mpi = np.array([d["t_comm"] for d in mpi_full])

    # 3. Speedup y eficiencia
    speedup = T_seq_full / T_total_mpi
    efficiency = speedup / ps

    print("\nTabla speedup/eficiencia (frac=1.0):")
    print("p\tT_total\tSpeedup\tEficiencia")
    for i, p in enumerate(ps):
        print(
            f"{p}\t{T_total_mpi[i]:.6f}\t{speedup[i]:.3f}\t{efficiency[i]:.3f}"
        )

    # 4. FLOPs/s en región de cómputo (misma flops_total que la secuencial)
    flops_per_sec = flops_total / T_comp_mpi

    print("\nFLOPs/s (región de cómputo de distancias, frac=1.0):")
    for i, p in enumerate(ps):
        print(f"p={p}, FLOPs/s={flops_per_sec[i]:.3e}")

    out_dir = Path("results/figures/mpi")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Gráficos ---

    # 1) Speedup vs p
    plt.figure()
    plt.plot(ps, speedup, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Speedup")
    plt.title("MPI KNN (strong scaling): Speedup vs p")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "mpi_speedup_vs_p.png", dpi=300)
    plt.close()

    # 2) Eficiencia vs p
    plt.figure()
    plt.plot(ps, efficiency, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Eficiencia")
    plt.title("MPI KNN (strong scaling): Eficiencia vs p")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "mpi_efficiency_vs_p.png", dpi=300)
    plt.close()

    # 3) Tiempos vs p
    plt.figure()
    plt.plot(ps, T_total_mpi, marker="o", label="T_total")
    plt.plot(ps, T_comp_mpi, marker="o", label="T_compute")
    plt.plot(ps, T_comm_mpi, marker="o", label="T_comm")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Tiempo (s)")
    plt.title("MPI KNN (strong scaling): Tiempos vs p (frac=1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mpi_times_vs_p.png", dpi=300)
    plt.close()

    # 4) FLOPs/s vs p
    plt.figure()
    plt.plot(ps, flops_per_sec, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("FLOPs/s")
    plt.title("MPI KNN (strong scaling): FLOPs/s vs p")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "mpi_flops_vs_p.png", dpi=300)
    plt.close()

    print("\nSe generaron los archivos en results/figures/mpi:")
    print("- mpi_speedup_vs_p.png")
    print("- mpi_efficiency_vs_p.png")
    print("- mpi_times_vs_p.png")
    print("- mpi_flops_vs_p.png")


if __name__ == "__main__":
    main()
