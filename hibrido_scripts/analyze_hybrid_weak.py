# analyze_hybrid_weak.py
import csv

import numpy as np
import matplotlib.pyplot as plt


def load_hybrid_weak(path: str):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) != 1:
        raise ValueError(f"Se esperaba 1 fila en {path}, hay {len(rows)}")

    r = rows[0]
    return {
        "p": int(r["p"]),
        "threads": int(r["threads"]),
        "W": int(r["W"]),
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
    # Mapea cada configuración (p,threads) a su archivo CSV weak
    files = {
        (1, 4): "hybrid_weak_p1_t4.csv",
        (2, 4): "hybrid_weak_p2_t4.csv",
        (4, 2): "hybrid_weak_p4_t2.csv",
        # añade más si hiciste más corridas
    }

    data = []
    for (p, t), path in files.items():
        d = load_hybrid_weak(path)
        data.append(d)

    # Ordenamos por W = p * threads
    data = sorted(data, key=lambda x: x["W"])

    W = np.array([d["W"] for d in data])
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
            f"{d['p']}\t{d['threads']}\t{d['W']}\t{d['frac']:.2f}\t"
            f"{d['workload']}\t{d['T_total']:.6f}\t{d['time_per_pair']:.3e}"
        )

    # ------- Gráfico 1: T_total vs W -------
    plt.figure()
    plt.plot(W, T_total, marker="o")
    for i in range(len(W)):
        plt.text(
            W[i] * 1.01,
            T_total[i],
            f"({ps[i]},{ts[i]})",
            fontsize=8,
        )
    plt.xlabel("Recursos totales W = p * threads")
    plt.ylabel("Tiempo total (s)")
    plt.title("Hybrid weak scaling: T_total vs W")
    plt.grid(True)
    plt.savefig("hybrid_weak_Ttotal_vs_W.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ------- Gráfico 2: tiempo normalizado vs W -------
    plt.figure()
    plt.plot(W, time_per_pair, marker="o")
    for i in range(len(W)):
        plt.text(
            W[i] * 1.01,
            time_per_pair[i],
            f"({ps[i]},{ts[i]})",
            fontsize=8,
        )
    plt.xlabel("Recursos totales W = p * threads")
    plt.ylabel("Tiempo por par train-test (s)")
    plt.title("Hybrid weak scaling: tiempo normalizado vs W")
    plt.grid(True)
    plt.savefig("hybrid_weak_time_per_pair_vs_W.png",
                dpi=300, bbox_inches="tight")
    plt.close()

    print("\nSe generaron:")
    print("- hybrid_weak_Ttotal_vs_W.png")
    print("- hybrid_weak_time_per_pair_vs_W.png")


if __name__ == "__main__":
    main()
