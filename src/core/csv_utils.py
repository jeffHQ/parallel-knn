# src/core/csv_utils.py

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Optional


CSV_COLUMNS = [
    "version",     
    "scaling",     
    "frac",       
    "n_train",
    "n_test",
    "k",
    "p",        
    "threads",   
    "workers",    
    "accuracy",
    "t_total",
    "t_compute",
    "t_comm",
    "flops",   
]


def append_result_row(
    path: Path | str,
    result: Dict,
    extra: Optional[Dict] = None,
    columns: Optional[Iterable[str]] = None,
) -> None:
    
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
