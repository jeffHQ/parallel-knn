# src/experiments/hybrid/analyze_hybrid_weak.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_hybrid_weak(path: Path):
    data = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["version"] != "hybrid":
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
                    "threads": int(row["threads"]),
                    "workers": int(row["workers"]),
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

    return sorted(data, key=lambda d: d["workers"])


def main() -> None:
    hyb_weak_path = Path("results/hybrid/hybrid_weak.csv")
    if not hyb_weak_path.exists():
        print(f"[ERROR] No se encontró {hyb_weak_path}. Corre antes experiments_hybrid_weak.py con varios (p,threads).")
        return

    data = load_hybrid_weak(hyb_weak_path)
    if not data:
        print("[ERROR] No se encontraron filas híbridas weak en hybrid_weak.csv.")
        return

    workers = np.array([d["workers"] for d in data])
    ps = np.array([d["p"] for d in data])
    ts = np.array([d["threads"] for d in data])
    fracs = np.array([d["frac"] for d in data])
    T_total = np.array([d["T_total"] for d in data])
    workload = np.array([d["workload"] for d in data])
    time_per_pair = np.array([d["time_per_pair"] for d in data])

    print("Tabla hybrid weak scaling:")
    print("p\tthreads\tW\tfrac\tworkload\tT_total\ttime_per_pair")
    for d in data:
        print(
            f"{d['p']}\t{d['threads']}\t{d['workers']}\t{d['frac']:.2f}\t"
            f"{d['workload']}\t{d['T_total']:.6f}\t{d['time_per_pair']:.3e}"
        )

    out_dir = Path("results/figures/hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) T_total vs workers
    plt.figure()
    plt.plot(workers, T_total, marker="o")
    for i in range(len(workers)):
        plt.text(workers[i] * 1.01, T_total[i], f"({ps[i]},{ts[i]})", fontsize=8)
    plt.xlabel("Recursos totales (workers = p * threads)")
    plt.ylabel("Tiempo total (s)")
    plt.title("Hybrid KNN (weak scaling): T_total vs workers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_weak_Ttotal_vs_workers.png", dpi=300)
    plt.close()

    # 2) Tiempo normalizado vs workers
    plt.figure()
    plt.plot(workers, time_per_pair, marker="o")
    for i in range(len(workers)):
        plt.text(
            workers[i] * 1.01,
            time_per_pair[i],
            f"({ps[i]},{ts[i]})",
            fontsize=8,
        )
    plt.xlabel("Recursos totales (workers = p * threads)")
    plt.ylabel("Tiempo por par train-test (s)")
    plt.title("Hybrid KNN (weak scaling): tiempo normalizado vs workers")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "hybrid_weak_time_per_pair_vs_workers.png", dpi=300)
    plt.close()

    print("\nSe generaron en results/figures/hybrid:")
    print("- hybrid_weak_Ttotal_vs_workers.png")
    print("- hybrid_weak_time_per_pair_vs_workers.png")


if __name__ == "__main__":
    main()
