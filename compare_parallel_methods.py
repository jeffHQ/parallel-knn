# compare_parallel_methods.py
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


# ---------- Utils de lectura ----------

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


def load_mpi_strong(path: str):
    """
    Lee un mpi_pX.csv con encabezado:
    'Resultados MPI (p = ...)'
    'frac,n_train,...'
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # quitar línea tipo "Resultados MPI (p = ...)"
    lines = [ln for ln in lines if not ln.startswith("Resultados MPI")]

    reader = csv.DictReader(lines)
    rows = list(reader)

    data = []
    for r in rows:
        data.append(
            {
                "frac": float(r["frac"]),
                "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]),
                "k": int(r["k"]),
                "acc": float(r["acc"]),
                "T_total": float(r["T_total"]),
                "T_comp": float(r["T_comp"]),
                "T_comm": float(r["T_comm"]),
            }
        )
    return data


def load_omp_strong(path: str):
    """
    Lee un omp_tX.csv con encabezado:
    'Resultados OMP (threads = ...)'
    'frac,n_train,...'
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # quitar línea tipo "Resultados OMP (threads = ...)"
    lines = [ln for ln in lines if not ln.startswith("Resultados OMP")]

    reader = csv.DictReader(lines)
    rows = list(reader)

    data = []
    for r in rows:
        data.append(
            {
                "frac": float(r["frac"]),
                "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]),
                "k": int(r["k"]),
                "acc": float(r["acc"]),
                "T_total": float(r["T_total"]),
            }
        )
    return data


def load_hybrid_strong(path: str):
    """
    Lee un hybrid_pX_tY.csv con encabezado:
    'Resultados Hybrid (p = ..., threads = ...)'
    'frac,n_train,...,T_total,T_comp,T_comm,p,threads'
    """
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    lines = [ln for ln in lines if not ln.startswith("Resultados Hybrid")]

    reader = csv.DictReader(lines)
    rows = list(reader)

    data = []
    for r in rows:
        data.append(
            {
                "frac": float(r["frac"]),
                "n_train": int(r["n_train"]),
                "n_test": int(r["n_test"]),
                "k": int(r["k"]),
                "acc": float(r["acc"]),
                "T_total": float(r["T_total"]),
                "T_comp": float(r["T_comp"]),
                "T_comm": float(r["T_comm"]),
                "p": int(r["p"]),
                "threads": int(r["threads"]),
            }
        )
    return data


# ---------- Función principal ----------

def main():
    # Un poco de estilo para que los plots se vean presentables
    plt.style.use("seaborn-v0_8")

    # 1) Cargar secuencial (baseline)
    seq_path = Path("seq_results/seq.csv")
    if not seq_path.exists():
        raise FileNotFoundError("No se encuentra seq.csv en la carpeta raíz.")
    seq_data = load_seq_results(str(seq_path))
    seq_full = next(d for d in seq_data if abs(d["frac"] - 1.0) < 1e-6)
    T_seq_full = seq_full["T_total"]
    n_train_full = seq_full["n_train"]
    n_test_full = seq_full["n_test"]

    print("Baseline secuencial (frac = 1.0):", seq_full)

    # 2) Configurar rutas de resultados strong scaling
    # Ajusta estos diccionarios si tus nombres de archivo son distintos
    mpi_strong_files = {
        1: "mpi_results/mpi_p1.csv",
        2: "mpi_results/mpi_p2.csv",
        4: "mpi_results/mpi_p4.csv",
        8: "mpi_results/mpi_p8.csv",
    }

    omp_strong_files = {
        1: "omp_results/omp_t1.csv",
        2: "omp_results/omp_t2.csv",
        4: "omp_results/omp_t4.csv",
        8: "omp_results/omp_t8.csv",
    }

    hybrid_strong_files = {
    (1, 2): "hibrido_results/hybrid_p1_t2.csv",
    (2, 2): "hibrido_results/hybrid_p2_t2.csv",
    (4, 2): "hibrido_results/hybrid_p4_t2.csv",
}


    # 3) Cargar MPI (frac = 1.0)
    mpi_configs = []
    for p, path in mpi_strong_files.items():
        if not Path(path).exists():
            print(f"[WARN] No se encontró {path}, se omite p={p}")
            continue
        data_p = load_mpi_strong(path)
        full_p = next(d for d in data_p if abs(d["frac"] - 1.0) < 1e-6)
        full_p["p"] = p
        mpi_configs.append(full_p)

    mpi_configs = sorted(mpi_configs, key=lambda d: d["p"])

    # 4) Cargar OMP (frac = 1.0)
    omp_configs = []
    for t, path in omp_strong_files.items():
        if not Path(path).exists():
            print(f"[WARN] No se encontró {path}, se omite threads={t}")
            continue
        data_t = load_omp_strong(path)
        full_t = next(d for d in data_t if abs(d["frac"] - 1.0) < 1e-6)
        full_t["threads"] = t
        omp_configs.append(full_t)

    omp_configs = sorted(omp_configs, key=lambda d: d["threads"])

    # 5) Cargar híbrido (frac = 1.0)
    hybrid_configs = []
    for (p, t), path in hybrid_strong_files.items():
        if not Path(path).exists():
            print(f"[WARN] No se encontró {path}, se omite (p={p},threads={t})")
            continue
        data_pt = load_hybrid_strong(path)
        full_pt = next(d for d in data_pt if abs(d["frac"] - 1.0) < 1e-6)
        hybrid_configs.append(full_pt)

    hybrid_configs = sorted(hybrid_configs, key=lambda d: d["p"] * d["threads"])

    # 6) Construir vectores para comparación
    #    W = recursos totales "tipo core lógico"
    #    MPI: W = p
    #    OMP: W = threads
    #    Hybrid: W = p * threads

    # --- MPI ---
    W_mpi = np.array([cfg["p"] for cfg in mpi_configs], dtype=int)
    T_mpi = np.array([cfg["T_total"] for cfg in mpi_configs])
    T_mpi_comp = np.array([cfg["T_comp"] for cfg in mpi_configs])
    T_mpi_comm = np.array([cfg["T_comm"] for cfg in mpi_configs])
    speedup_mpi = T_seq_full / T_mpi
    eff_mpi = speedup_mpi / W_mpi

    # --- OMP ---
    W_omp = np.array([cfg["threads"] for cfg in omp_configs], dtype=int)
    T_omp = np.array([cfg["T_total"] for cfg in omp_configs])
    speedup_omp = T_seq_full / T_omp
    eff_omp = speedup_omp / W_omp

    # --- Hybrid ---
    W_hyb = np.array([cfg["p"] * cfg["threads"] for cfg in hybrid_configs], dtype=int)
    T_hyb = np.array([cfg["T_total"] for cfg in hybrid_configs])
    T_hyb_comp = np.array([cfg["T_comp"] for cfg in hybrid_configs])
    T_hyb_comm = np.array([cfg["T_comm"] for cfg in hybrid_configs])
    speedup_hyb = T_seq_full / T_hyb
    eff_hyb = speedup_hyb / W_hyb

    # 7) FLOPs/s efectivos (usando T_total para comparar justo)
    digits = load_digits()
    d = digits.data.shape[1]  # 64
    flops_per_distance = 3 * d + 1
    flops_total = n_train_full * n_test_full * flops_per_distance

    flops_mpi = flops_total / T_mpi
    flops_omp = flops_total / T_omp
    flops_hyb = flops_total / T_hyb

    # 8) Resumen textual en consola
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
    print("p\tthreads\tW\tT_total\tSpeedup\tEficiencia\tFLOPs/s")
    for i, cfg in enumerate(hybrid_configs):
        print(
            f"{cfg['p']}\t{cfg['threads']}\t{W_hyb[i]}\t"
            f"{T_hyb[i]:.6f}\t{speedup_hyb[i]:.3f}\t{eff_hyb[i]:.3f}\t"
            f"{flops_hyb[i]:.3e}"
        )

    # 9) Gráficos comparativos

    # 9.1 Speedup vs W
    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, speedup_mpi, "o-", label="MPI")
    plt.plot(W_omp, speedup_omp, "s-", label="OMP")
    plt.plot(W_hyb, speedup_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W (procesos / hilos efectivos)")
    plt.ylabel("Speedup respecto al secuencial")
    plt.title("Comparación de speedup: MPI vs OMP vs Híbrido (frac=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_speedup_all.png", dpi=300)

    # 9.2 Eficiencia vs W
    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, eff_mpi, "o-", label="MPI")
    plt.plot(W_omp, eff_omp, "s-", label="OMP")
    plt.plot(W_hyb, eff_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W")
    plt.ylabel("Eficiencia")
    plt.title("Comparación de eficiencia: MPI vs OMP vs Híbrido (frac=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_efficiency_all.png", dpi=300)

    # 9.3 Tiempo total vs W
    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, T_mpi, "o-", label="MPI")
    plt.plot(W_omp, T_omp, "s-", label="OMP")
    plt.plot(W_hyb, T_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W")
    plt.ylabel("Tiempo total (s)")
    plt.title("Tiempo total vs recursos: MPI vs OMP vs Híbrido (frac=1.0)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_time_all.png", dpi=300)

    # 9.4 FLOPs/s vs W
    plt.figure(figsize=(7, 5))
    plt.plot(W_mpi, flops_mpi, "o-", label="MPI")
    plt.plot(W_omp, flops_omp, "s-", label="OMP")
    plt.plot(W_hyb, flops_hyb, "^-", label="Híbrido")
    plt.xlabel("Recursos totales W")
    plt.ylabel("FLOPs/s efectivos")
    plt.title("Rendimiento efectivo (FLOPs/s): MPI vs OMP vs Híbrido")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("compare_flops_all.png", dpi=300)

    print("\nSe generaron las figuras:")
    print("- compare_speedup_all.png")
    print("- compare_efficiency_all.png")
    print("- compare_time_all.png")
    print("- compare_flops_all.png")


if __name__ == "__main__":
    main()
