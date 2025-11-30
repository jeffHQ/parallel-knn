# analyze_mpi_weak.py
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_weak_file(path: str):
    with open(path, "r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # Quitamos la línea "Resultados weak scaling (p = ...)"
    lines = [ln for ln in lines if not ln.startswith("Resultados weak")]

    reader = csv.DictReader(lines)
    rows = list(reader)
    if len(rows) != 1:
        raise ValueError(f"Se esperaba 1 fila en {path}, hay {len(rows)}")

    r = rows[0]
    return {
        "p": int(r["p"]),
        "frac": float(r["frac"]),
        "n_train": int(r["n_train"]),
        "n_test": int(r["n_test"]),
        "k": int(r["k"]),
        "acc": float(r["acc"]),
        "T_total": float(r["T_total"]),
        "T_comp": float(r["T_comp"]),
        "T_comm": float(r["T_comm"]),
        "workload": int(r["workload"]),
        "time_per_pair": float(r["time_per_pair"]),
    }


def main():
    files = {
        1: "weak_p1.csv",
        2: "weak_p2.csv",
        4: "weak_p4.csv",
        8: "weak_p8.csv",
    }

    data = []
    for p, path in files.items():
        d = load_weak_file(path)
        data.append(d)

    data = sorted(data, key=lambda x: x["p"])

    ps = np.array([d["p"] for d in data])
    fracs = np.array([d["frac"] for d in data])
    T_total = np.array([d["T_total"] for d in data])
    workload = np.array([d["workload"] for d in data])
    time_per_pair = np.array([d["time_per_pair"] for d in data])

    print("Tabla weak scaling:")
    print("p\tfrac\tworkload\tT_total\ttime_per_pair")
    for d in data:
        print(
            f"{d['p']}\t{d['frac']:.2f}\t{d['workload']}\t"
            f"{d['T_total']:.6f}\t{d['time_per_pair']:.3e}"
        )

    # Gráfico 1: T_total vs p
    plt.figure()
    plt.plot(ps, T_total, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Tiempo total (s)")
    plt.title("Weak scaling: T_total vs p")
    plt.grid(True)
    plt.savefig("weak_Ttotal_vs_p.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico 2: tiempo por par (trabajo normalizado) vs p
    plt.figure()
    plt.plot(ps, time_per_pair, marker="o")
    plt.xlabel("Número de procesos (p)")
    plt.ylabel("Tiempo por par train-test (s)")
    plt.title("Weak scaling: tiempo normalizado vs p")
    plt.grid(True)
    plt.savefig("weak_time_per_pair_vs_p.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSe generaron:")
    print("- weak_Ttotal_vs_p.png")
    print("- weak_time_per_pair_vs_p.png")


if __name__ == "__main__":
    main()
