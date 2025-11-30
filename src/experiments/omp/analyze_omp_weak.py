# src/experiments/omp/analyze_omp_weak.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_omp_weak(path: Path):
    data = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["version"] != "omp":
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
                    "threads": int(row["threads"]),
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

    return sorted(data, key=lambda d: d["threads"])


def main() -> None:
    omp_weak_path = Path("results/omp/omp_weak.csv")
    if not omp_weak_path.exists():
        print(f"[ERROR] No se encontró {omp_weak_path}. Corre antes experiments_omp_weak.py.")
        return

    data = load_omp_weak(omp_weak_path)
    if not data:
        print("[ERROR] No se encontraron filas OMP weak en omp_weak.csv.")
        return

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

    out_dir = Path("results/figures/omp")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) T_total vs hilos
    plt.figure()
    plt.plot(threads, T_total, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Tiempo total (s)")
    plt.title("OMP KNN (weak scaling): T_total vs hilos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "omp_weak_Ttotal_vs_threads.png", dpi=300)
    plt.close()

    # 2) Tiempo normalizado por par train-test vs hilos
    plt.figure()
    plt.plot(threads, time_per_pair, marker="o")
    plt.xlabel("Número de hilos")
    plt.ylabel("Tiempo por par train-test (s)")
    plt.title("OMP KNN (weak scaling): tiempo normalizado vs hilos")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "omp_weak_time_per_pair_vs_threads.png", dpi=300)
    plt.close()

    print("\nSe generaron en results/figures/omp:")
    print("- omp_weak_Ttotal_vs_threads.png")
    print("- omp_weak_time_per_pair_vs_threads.png")


if __name__ == "__main__":
    main()
