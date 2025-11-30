# src/core/csv_utils.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional


# Esquema "amplio" que te sirve para seq, MPI, OMP e híbrido.
# No tienes que usar todos los campos siempre; se pueden dejar vacíos.
CSV_COLUMNS = [
    "version",     # "sequential", "mpi", "omp", "hybrid"
    "scaling",     # "strong" o "weak"
    "frac",        # fracción del dataset (0.0–1.0)
    "n_train",
    "n_test",
    "k",
    "p",           # procesos MPI
    "threads",     # hilos (joblib)
    "workers",     # p * threads
    "accuracy",
    "t_total",
    "t_compute",
    "t_comm",
    "flops",       # FLOPs totales de la región de cómputo (opcional)
]


def append_result_row(
    path: Path | str,
    result: Dict,
    extra: Optional[Dict] = None,
    columns: Optional[Iterable[str]] = None,
) -> None:
    """
    Añade una fila de resultados a un CSV, creando el archivo y el
    encabezado si no existen.

    - result: dict que típicamente viene de run_sequential/run_mpi/...,
      con claves como "n_train", "n_test", "k", "accuracy", "t_total", etc.
    - extra: dict opcional con metadatos adicionales, por ejemplo
      {"version": "mpi", "scaling": "strong", "workers": p*threads}.
    - columns: lista de columnas a usar; si no se da, se usa CSV_COLUMNS.
    """
    if columns is None:
        columns = CSV_COLUMNS

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    row_data: Dict[str, object] = {}
    extra = extra or {}

    for col in columns:
        if col in result:
            row_data[col] = result[col]
        elif col in extra:
            row_data[col] = extra[col]
        else:
            row_data[col] = ""

    file_exists = path.exists()

    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)
