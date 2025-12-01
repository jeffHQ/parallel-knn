# src/experiments/compare_parallel_methods.py

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _load_generic_results(path: Path) -> list[dict]:
  
    data: list[dict] = []

    if not path.exists():
        print(f"[WARN] No se encontró {path}")
        return data

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
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
            except KeyError as exc:
                raise KeyError(
                    f"Falta columna {exc} en {path}. "
                    f"Revisa que los CSV tengan el formato unificado."
                ) from exc

    return data


def load_seq_strong(path: Path) -> dict:
    
    data = _load_generic_results(path)
    data = [
        d for d in data
        if d["version"] == "sequential" and d["scaling"] == "strong"
    ]
    if not data:
        raise RuntimeError(
            f"No se encontraron filas 'sequential,strong' en {path}"
        )

   
    full = [d for d in data if abs(d["frac"] - 1.0) < 1e-6]
    if not full:
        raise RuntimeError(
            f"No se encontró fila secuencial con frac=1.0 en {path}"
        )


    return full[0]


def load_method_strong(path: Path, version: str) -> list[dict]:

    data = _load_generic_results(path)
    data = [
        d for d in data
        if d["version"] == version and d["scaling"] == "strong"
    ]
    return data



def main() -> None:
   
    plt.style.use("seaborn-v0_8")

    base_dir = Path("results")

    seq_path = base_dir / "seq" / "seq.csv"
    mpi_path = base_dir / "mpi" / "mpi_strong.csv"
    omp_path = base_dir / "omp" / "omp_strong.csv"
    hyb_path = base_dir / "hybrid" / "hybrid_strong.csv"

    out_dir = base_dir / "figures" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)


    if not seq_path.exists():
        print(f"[ERROR] No se encontró {seq_path}. Corre primero experiments_seq.")
        return

    seq_full = load_seq_strong(seq_path)
    T_seq_full = seq_full["t_total"]
    n_train_full = seq_full["n_train"]
    n_test_full = seq_full["n_test"]
    flops_total = seq_full["flops"]

    print("Baseline secuencial (frac=1.0):")
    print(seq_full)

    if not mpi_path.exists():
        print(f"[ERROR] No se encontró {mpi_path}. Corre antes experiments_mpi_strong.")
        return

    mpi_all = load_method_strong(mpi_path, version="mpi")
    mpi_full = [d for d in mpi_all if abs(d["frac"] - 1.0) < 1e-6]

    if not mpi_full:
        print(f"[ERROR] No hay filas MPI con frac=1.0 en {mpi_path}.")
        return

  
    mpi_full.sort(key=lambda d: d["workers"])

    W_mpi = np.array([d["workers"] for d in mpi_full], dtype=int)
    T_mpi = np.array([d["t_total"] for d in mpi_full])
    T_mpi_comp = np.array([d["t_compute"] for d in mpi_full])
    T_mpi_comm = np.array([d["t_comm"] for d in mpi_full])

    speedup_mpi = T_seq_full / T_mpi
    eff_mpi = speedup_mpi / W_mpi


    flops_mpi = flops_total / T_mpi


    if not omp_path.exists():
        print(f"[ERROR] No se encontró {omp_path}. Corre antes experiments_omp_strong.")
        return

    omp_all = load_method_strong(omp_path, version="omp")
    omp_full = [d for d in omp_all if abs(d["frac"] - 1.0) < 1e-6]

    if not omp_full:
        print(f"[ERROR] No hay filas OMP con frac=1.0 en {omp_path}.")
        return

    # workers = threads (p=1)
    omp_full.sort(key=lambda d: d["workers"])

    W_omp = np.array([d["workers"] for d in omp_full], dtype=int)
    T_omp = np.array([d["t_total"] for d in omp_full])

    speedup_omp = T_seq_full / T_omp
    eff_omp = speedup_omp / W_omp
    flops_omp = flops_total / T_omp


    if not hyb_path.exists():
        print(f"[ERROR] No se encontró {hyb_path}. Corre antes experiments_hybrid_strong.")
        return

    hyb_all = load_method_strong(hyb_path, version="hybrid")
    hyb_full = [d for d in hyb_all if abs(d["frac"] - 1.0) < 1e-6]

    if not hyb_full:
        print(f"[ERROR] No hay filas HYBRID con frac=1.0 en {hyb_path}.")
        return


    hyb_full.sort(key=lambda d: d["workers"])

    W_hyb = np.array([d["workers"] for d in hyb_full], dtype=int)
    T_hyb = np.array([d["t_total"] for d in hyb_full])
    T_hyb_comp = np.array([d["t_compute"] for d in hyb_full])
    T_hyb_comm = np.array([d["t_comm"] for d in hyb_full])

    speedup_hyb = T_seq_full / T_hyb
    eff_hyb = speedup_hyb / W_hyb
    flops_hyb = flops_total / T_hyb



    print("\n== MPI (frac=1.0) ==")
    print("W=p\tT_total\tSpeedup\tEficiencia\tFLOPs/s")
    for i, W in enumerate(W_mpi):
        print(
            f"{W}\t{T_mpi[i]:.6f}\t"
            f"{speedup_mpi[i]:.3f}\t{eff_mpi[i]:.3f}\t{flops_mpi[i]:.3e}"
        )

    print("\n== OMP (frac=1.0) ==")
    print("W=threads\tT_total\tSpeedup\tEficiencia\tFLOPs/s")
    for i, W in enumerate(W_omp):
        print(
            f"{W}\t\t{T_omp[i]:.6f}\t"
            f"{speedup_omp[i]:.3f}\t{eff_omp[i]:.3f}\t{flops_omp[i]:.3e}"
        )

    print("\n== Híbrido (frac=1.0) ==")
    print("p\tthreads\tW=p*threads\tT_total\tSpeedup\tEficiencia\tFLOPs/s")
    for i, d in enumerate(hyb_full):
        print(
            f"{d['p']}\t{d['threads']}\t{W_hyb[i]}\t"
            f"{T_hyb[i]:.6f}\t{speedup_hyb[i]:.3f}\t{eff_hyb[i]:.3f}\t"
            f"{flops_hyb[i]:.3e}"
        )




    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, speedup_mpi, "o-", label="MPI")
    plt.plot(W_omp, speedup_omp, "s-", label="OMP")
    plt.plot(W_hyb, speedup_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W (procesos / hilos efectivos)")
    plt.ylabel("Speedup respecto al secuencial")
    plt.title("Speedup: MPI vs OMP vs Híbrido (frac=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_speedup_all.png", dpi=300)
    plt.close()


    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, eff_mpi, "o-", label="MPI")
    plt.plot(W_omp, eff_omp, "s-", label="OMP")
    plt.plot(W_hyb, eff_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W")
    plt.ylabel("Eficiencia")
    plt.title("Eficiencia: MPI vs OMP vs Híbrido (frac=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_efficiency_all.png", dpi=300)
    plt.close()


    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, T_mpi, "o-", label="MPI")
    plt.plot(W_omp, T_omp, "s-", label="OMP")
    plt.plot(W_hyb, T_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W")
    plt.ylabel("Tiempo total (s)")
    plt.title("Tiempo total: MPI vs OMP vs Híbrido (frac=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_time_all.png", dpi=300)
    plt.close()


    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, flops_mpi, "o-", label="MPI")
    plt.plot(W_omp, flops_omp, "s-", label="OMP")
    plt.plot(W_hyb, flops_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W")
    plt.ylabel("FLOPs/s efectivos")
    plt.title("FLOPs/s efectivos: MPI vs OMP vs Híbrido (frac=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "compare_flops_all.png", dpi=300)
    plt.close()

    print("\nFiguras generadas en results/figures/compare:")
    print("- compare_speedup_all.png")
    print("- compare_efficiency_all.png")
    print("- compare_time_all.png")
    print("- compare_flops_all.png")


if __name__ == "__main__":
    main()
