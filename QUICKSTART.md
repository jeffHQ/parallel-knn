# QUICK START GUIDE - Parallel KNN

## Setup Inicial (Solo una vez)

```powershell
# 1. Clonar repositorio
cd C:\Users\TuUsuario\Documents
git clone https://github.com/jeffHQ/parallel-knn.git
cd parallel-knn

# 2. Crear entorno virtual
python -m venv .venv

# 3. Activar entorno (PowerShell)
.\.venv\Scripts\Activate.ps1

# 4. Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 5. Verificar instalación
python -c "from mpi4py import MPI; print('OK')"
mpiexec -n 2 python -c "from mpi4py import MPI; print(f'Rank {MPI.COMM_WORLD.rank}')"
```

---

## Ejecutar TODOS los Experimentos (Automático)

```powershell
# Navegar al proyecto
cd C:\Users\TuUsuario\Documents\parallel-knn

# Activar entorno
.\.venv\Scripts\Activate.ps1

# Ejecutar script automático (puede tardar 30-60 minutos)
.\run_all_experiments.ps1
```

---

## Generar TODAS las Gráficas (Automático)

```powershell
# Después de ejecutar experimentos
.\generate_all_plots.ps1
```

---

## Comandos Individuales (Manual)

### 1. Secuencial
```powershell
cd src
python experiments/seq/experiments_seq.py --clear
python experiments/seq/analyze_seq.py
```

### 2. MPI Strong Scaling (W = 1, 2, 4, 8, 16)
```powershell
cd src
mpiexec -n 1 python experiments/mpi/experiments_mpi_strong.py --clear
mpiexec -n 2 python experiments/mpi/experiments_mpi_strong.py
mpiexec -n 4 python experiments/mpi/experiments_mpi_strong.py
mpiexec -n 8 python experiments/mpi/experiments_mpi_strong.py
mpiexec -n 16 python experiments/mpi/experiments_mpi_strong.py

# Generar gráficas
python experiments/mpi/analyze_mpi_strong.py
```

### 3. MPI Weak Scaling
```powershell
cd src
mpiexec -n 1 python experiments/mpi/experiments_mpi_weak.py --clear
mpiexec -n 2 python experiments/mpi/experiments_mpi_weak.py
mpiexec -n 4 python experiments/mpi/experiments_mpi_weak.py
mpiexec -n 8 python experiments/mpi/experiments_mpi_weak.py
mpiexec -n 16 python experiments/mpi/experiments_mpi_weak.py

# Generar gráficas
python experiments/mpi/analyze_mpi_weak.py
```

### 4. OMP Strong Scaling (Threads = 1, 2, 4, 8, 16)
```powershell
cd src
python experiments/omp/experiments_omp_strong.py --clear --threads-list 1 2 4 8 16

# Generar gráficas
python experiments/omp/analyze_omp_strong.py
```

### 5. OMP Weak Scaling
```powershell
cd src
python experiments/omp/experiments_omp_weak.py --clear --threads-list 1 2 4 8 16

# Generar gráficas
python experiments/omp/analyze_omp_weak.py
```

### 6. Hybrid Strong Scaling (W = p × threads)
```powershell
cd src
# W = 1
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_strong.py --clear --threads 1

# W = 2
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_strong.py --threads 2

# W = 4
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_strong.py --threads 2

# W = 8
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_strong.py --threads 4

# W = 16 (varias opciones)
mpiexec -n 4 python experiments/hybrid/experiments_hybrid_strong.py --threads 4
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_strong.py --threads 8
mpiexec -n 8 python experiments/hybrid/experiments_hybrid_strong.py --threads 2

# Generar gráficas
python experiments/hybrid/analyze_hybrid_strong.py
```

### 7. Hybrid Weak Scaling
```powershell
cd src
# W = 1
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_weak.py --clear --threads 1

# W = 2
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_weak.py --threads 2

# W = 4
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_weak.py --threads 2

# W = 8
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_weak.py --threads 4

# W = 16
mpiexec -n 4 python experiments/hybrid/experiments_hybrid_weak.py --threads 4

# Generar gráficas
python experiments/hybrid/analyze_hybrid_weak.py
```

### 8. Comparación Global (MPI vs OMP vs Hybrid)
```powershell
cd src
python experiments/compare/compare_parallel_methods.py
```

---

## Ver Ayuda de Cualquier Script

```powershell
python experiments/seq/experiments_seq.py --help
python experiments/omp/experiments_omp_strong.py --help
mpiexec -n 1 python experiments/mpi/experiments_mpi_strong.py --help
```

---

## Estructura de Resultados

```
results/
├── seq/
│   └── seq.csv
├── mpi/
│   ├── mpi_strong.csv
│   └── mpi_weak.csv
├── omp/
│   ├── omp_strong.csv
│   └── omp_weak.csv
├── hybrid/
│   ├── hybrid_strong.csv
│   └── hybrid_weak.csv
└── figures/
    ├── seq/           (2 PNG)
    ├── mpi/           (6 PNG)
    ├── omp/           (6 PNG)
    ├── hybrid/        (6 PNG)
    └── compare/       (4 PNG - comparación global)
```

---

## Troubleshooting Rápido

### Error: "mpiexec no reconocido"
```powershell
# Reinstalar MS-MPI y reiniciar terminal
# O agregar al PATH manualmente:
$env:Path += ";C:\Program Files\Microsoft MPI\Bin"
```

### Error: "Import mpi4py could not be resolved"
```powershell
pip uninstall mpi4py
pip install mpi4py
```

### Limpiar y reiniciar experimentos
```powershell
# Borra todos los CSV
Remove-Item results\*\*.csv

# O usa --clear en cada experimento
python experiments/seq/experiments_seq.py --clear
```

---

## Tiempos Estimados (en máquina con 16 cores)

- **Secuencial**: ~2-5 min
- **MPI Strong** (5 configuraciones): ~10-15 min
- **MPI Weak** (5 configuraciones): ~10-15 min
- **OMP Strong**: ~10-15 min
- **OMP Weak**: ~10-15 min
- **Hybrid Strong** (7 configuraciones): ~15-20 min
- **Hybrid Weak** (5 configuraciones): ~10-15 min
- **Generar todas las gráficas**: ~1-2 min

**TOTAL**: ~60-90 minutos para ejecutar todo

---

## Configuraciones Recomendadas para Análisis

### Para Strong Scaling (comparar con mismo problema):
- Usa siempre `frac=1.0` (dataset completo)
- Varia W: 1, 2, 4, 8, 16
- MPI: varia p
- OMP: varia threads
- Hybrid: varias combinaciones (p, threads) para mismo W

### Para Weak Scaling (problema crece con W):
- `base_frac=0.25` (estándar)
- Dataset crece proporcionalmente: W=1→0.25, W=2→0.5, W=4→1.0
- Tiempo debería mantenerse constante (ideal)

### Para Comparación Global:
- Solo usa experimentos strong con `frac=1.0`
- Asegúrate de tener: seq, mpi_strong, omp_strong, hybrid_strong

---

## Archivos Clave del Proyecto

- **README.md**: Documentación completa
- **QUICKSTART.md**: Esta guía rápida
- **requirements.txt**: Dependencias Python
- **run_all_experiments.ps1**: Script automático de experimentos
- **generate_all_plots.ps1**: Script automático de gráficas
- **src/methods/**: Implementaciones KNN (sequential, mpi, omp, hybrid)
- **src/experiments/**: Scripts de experimentos y análisis
- **results/**: CSV y gráficas generadas
