# src/experiments/mpi/analyze_mpi_weak.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_mpi_weak(path: Path):
    data = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["version"] != "mpi":
                continue
            if row["scaling"] != "weak":
                continue
            n_train = int(row["n_train"])
            n_test = int(row["n_test"])
            T_total = float(row["t_total"])
            workload = n_train * n_test
            time_per_pair = T_total / workload if workload > 0 else 0.0

            data.append(
                {
                    "p": int(row["p"]),
                    "frac": float(row["frac"]),
                    "n_train": n_train,
                    "n_test": n_test,
                    "accuracy": float(row["accuracy"]),
                    "T_total": T_total,
                    "T_compute": float(row["t_compute"]),
                    "T_comm": float(row["t_comm"]),
                    "workload": workload,
                    "time_per_pair": time_per_pair,
                }
            )
    return data


def main() -> None:
    mpi_weak_path = Path("results/mpi/mpi_weak.csv")
    if not mpi_weak_path.exists():
        print(f"[ERROR] No se encontró {mpi_weak_path}. Corre antes experiments_mpi_weak.py con varios p.")
        return

    data = load_mpi_weak(mpi_weak_path)
    if not data:
        print("[ERROR] No se encontraron filas weak en mpi_weak.csv.")
        return

    data = sorted(data, key=lambda d: d["p"])

    ps = np.array([d["p"] for d in data])
    fracs = np.array([d["frac"] for d in data])
    T_total = np.array([d["T_total"] for d in data])
    workload = np.array([d["workload"] for d in data])
    time_per_pair = np.array([d["time_per_pair"] for d in data])

    print("Tabla weak scaling (MPI):")
    print("p\tfrac\tworkload\tT_total\ttime_per_pair")
    for d in data:
        print(
            f"{d['p']}\t{d['frac']:.2f}\t{d['workload']}\t"
            f"{d['T_total']:.6f}\t{d['time_per_pair']:.3e}"
        )

    out_dir = Path("results/figures/mpi")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) T_total vs p
    plt.figure()
    plt.plot(ps, T_total, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Tiempo total (s)")
    plt.title("MPI KNN (weak scaling): T_total vs p")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "mpi_weak_Ttotal_vs_p.png", dpi=300)
    plt.close()

    # 2) Tiempo normalizado por par train-test vs p
    plt.figure()
    plt.plot(ps, time_per_pair, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Tiempo por par train-test (s)")
    plt.title("MPI KNN (weak scaling): tiempo normalizado vs p")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "mpi_weak_time_per_pair_vs_p.png", dpi=300)
    plt.close()

    print("\nSe generaron en results/figures/mpi:")
    print("- mpi_weak_Ttotal_vs_p.png")
    print("- mpi_weak_time_per_pair_vs_p.png")


if __name__ == "__main__":
    main()
