# src/experiments/seq/analyze_seq.py

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
                    "p": int(row["p"]),
                    "threads": int(row["threads"]),
                    "workers": int(row["workers"]),
                    "accuracy": float(row["accuracy"]),
                    "t_total": float(row["t_total"]),
                    "t_compute": float(row["t_compute"]),
                    "t_comm": float(row["t_comm"]),
                    "flops": float(row["flops"]) if row["flops"] else 0.0,
                }
            )
    return data


def main() -> None:
    results_path = Path("results/seq/seq.csv")
    if not results_path.exists():
        print(f"[ERROR] No se encontró {results_path}. Corre primero experiments_seq.py.")
        return

    data = load_seq_results(results_path)
   
    data = sorted(data, key=lambda d: d["frac"])

    fracs = np.array([d["frac"] for d in data])
    n_train = np.array([d["n_train"] for d in data])
    n_test = np.array([d["n_test"] for d in data])
    acc = np.array([d["accuracy"] for d in data])
    t_total = np.array([d["t_total"] for d in data])
    t_compute = np.array([d["t_compute"] for d in data])
    flops = np.array([d["flops"] for d in data])

    # Evitar división por cero
    flops_per_sec = np.where(t_compute > 0, flops / t_compute, 0.0)

    print("Tabla resultados secuenciales (strong scaling sobre tamaño de problema):")
    print("frac\tn_train\tn_test\tacc\tT_total\tFLOPs\tFLOPs/s")
    for d, fps in zip(data, flops_per_sec):
        print(
            f"{d['frac']:.2f}\t"
            f"{d['n_train']}\t"
            f"{d['n_test']}\t"
            f"{d['accuracy']:.4f}\t"
            f"{d['t_total']:.6f}\t"
            f"{d['flops']:.3e}\t"
            f"{fps:.3e}"
        )

    out_dir = Path("results/figures/seq")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Tiempo total vs fracción del dataset
    plt.figure()
    plt.plot(fracs, t_total, marker="o")
    plt.xlabel("Fracción del dataset (frac)")
    plt.ylabel("Tiempo total (s)")
    plt.title("KNN secuencial: T_total vs fracción del dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "seq_Ttotal_vs_frac.png", dpi=300)
    plt.close()

    # 2) Accuracy vs fracción del dataset
    plt.figure()
    plt.plot(fracs, acc, marker="o")
    plt.xlabel("Fracción del dataset (frac)")
    plt.ylabel("Accuracy")
    plt.title("KNN secuencial: accuracy vs fracción del dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "seq_accuracy_vs_frac.png", dpi=300)
    plt.close()

    # 3) FLOPs/s vs fracción del dataset
    plt.figure()
    plt.plot(fracs, flops_per_sec, marker="o")
    plt.xlabel("Fracción del dataset (frac)")
    plt.ylabel("FLOPs/s (estimado)")
    plt.title("KNN secuencial: FLOPs/s vs fracción del dataset")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "seq_flops_vs_frac.png", dpi=300)
    plt.close()

    print("\nSe generaron los gráficos en results/figures/seq:")
    print("- seq_Ttotal_vs_frac.png")
    print("- seq_accuracy_vs_frac.png")
    print("- seq_flops_vs_frac.png")


if __name__ == "__main__":
    main()
