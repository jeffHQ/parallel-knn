# run_all_experiments.ps1
# Script de PowerShell para ejecutar todos los experimentos en Windows
# 
# USO: 
#   .\run_all_experiments.ps1
#
# O ejecutar secciones específicas comentando/descomentando líneas

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Parallel KNN - Ejecutar Todos los Experimentos" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Cambiar al directorio src/
Set-Location src

# ============================================================
# 1. SECUENCIAL (Baseline)
# ============================================================
Write-Host "[1/7] Ejecutando experimentos SECUENCIAL..." -ForegroundColor Green
python experiments/seq/experiments_seq.py --clear
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en experimentos secuenciales" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/seq/seq.csv" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 2. MPI STRONG SCALING
# ============================================================
Write-Host "[2/7] Ejecutando experimentos MPI STRONG SCALING..." -ForegroundColor Green
Write-Host "    Ejecutando con p=1..." -ForegroundColor Gray
mpiexec -n 1 python experiments/mpi/experiments_mpi_strong.py --clear

Write-Host "    Ejecutando con p=2..." -ForegroundColor Gray
mpiexec -n 2 python experiments/mpi/experiments_mpi_strong.py

Write-Host "    Ejecutando con p=4..." -ForegroundColor Gray
mpiexec -n 4 python experiments/mpi/experiments_mpi_strong.py

Write-Host "    Ejecutando con p=8..." -ForegroundColor Gray
mpiexec -n 8 python experiments/mpi/experiments_mpi_strong.py

Write-Host "    Ejecutando con p=16..." -ForegroundColor Gray
mpiexec -n 16 python experiments/mpi/experiments_mpi_strong.py

Write-Host "    Completado: results/mpi/mpi_strong.csv" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 3. MPI WEAK SCALING
# ============================================================
Write-Host "[3/7] Ejecutando experimentos MPI WEAK SCALING..." -ForegroundColor Green
Write-Host "    Ejecutando con p=1..." -ForegroundColor Gray
mpiexec -n 1 python experiments/mpi/experiments_mpi_weak.py --clear

Write-Host "    Ejecutando con p=2..." -ForegroundColor Gray
mpiexec -n 2 python experiments/mpi/experiments_mpi_weak.py

Write-Host "    Ejecutando con p=4..." -ForegroundColor Gray
mpiexec -n 4 python experiments/mpi/experiments_mpi_weak.py

Write-Host "    Ejecutando con p=8..." -ForegroundColor Gray
mpiexec -n 8 python experiments/mpi/experiments_mpi_weak.py

Write-Host "    Ejecutando con p=16..." -ForegroundColor Gray
mpiexec -n 16 python experiments/mpi/experiments_mpi_weak.py

Write-Host "    Completado: results/mpi/mpi_weak.csv" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 4. OMP STRONG SCALING
# ============================================================
Write-Host "[4/7] Ejecutando experimentos OMP STRONG SCALING..." -ForegroundColor Green
python experiments/omp/experiments_omp_strong.py --clear --threads-list 1 2 4 8 16
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en experimentos OMP strong" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/omp/omp_strong.csv" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 5. OMP WEAK SCALING
# ============================================================
Write-Host "[5/7] Ejecutando experimentos OMP WEAK SCALING..." -ForegroundColor Green
python experiments/omp/experiments_omp_weak.py --clear --threads-list 1 2 4 8 16
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en experimentos OMP weak" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/omp/omp_weak.csv" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 6. HYBRID STRONG SCALING
# ============================================================
Write-Host "[6/7] Ejecutando experimentos HYBRID STRONG SCALING..." -ForegroundColor Green
Write-Host "    W=1: p=1, threads=1..." -ForegroundColor Gray
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_strong.py --clear --threads 1

Write-Host "    W=2: p=1, threads=2..." -ForegroundColor Gray
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_strong.py --threads 2

Write-Host "    W=4: p=2, threads=2..." -ForegroundColor Gray
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_strong.py --threads 2

Write-Host "    W=8: p=2, threads=4..." -ForegroundColor Gray
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_strong.py --threads 4

Write-Host "    W=16: p=4, threads=4..." -ForegroundColor Gray
mpiexec -n 4 python experiments/hybrid/experiments_hybrid_strong.py --threads 4

# Configuraciones adicionales para W=16 (puedes comentar si son muchas)
Write-Host "    W=16: p=2, threads=8..." -ForegroundColor Gray
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_strong.py --threads 8

Write-Host "    W=16: p=8, threads=2..." -ForegroundColor Gray
mpiexec -n 8 python experiments/hybrid/experiments_hybrid_strong.py --threads 2

Write-Host "    Completado: results/hybrid/hybrid_strong.csv" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 7. HYBRID WEAK SCALING
# ============================================================
Write-Host "[7/7] Ejecutando experimentos HYBRID WEAK SCALING..." -ForegroundColor Green
Write-Host "    W=1: p=1, threads=1..." -ForegroundColor Gray
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_weak.py --clear --threads 1

Write-Host "    W=2: p=1, threads=2..." -ForegroundColor Gray
mpiexec -n 1 python experiments/hybrid/experiments_hybrid_weak.py --threads 2

Write-Host "    W=4: p=2, threads=2..." -ForegroundColor Gray
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_weak.py --threads 2

Write-Host "    W=8: p=2, threads=4..." -ForegroundColor Gray
mpiexec -n 2 python experiments/hybrid/experiments_hybrid_weak.py --threads 4

Write-Host "    W=16: p=4, threads=4..." -ForegroundColor Gray
mpiexec -n 4 python experiments/hybrid/experiments_hybrid_weak.py --threads 4

Write-Host "    Completado: results/hybrid/hybrid_weak.csv" -ForegroundColor Gray
Write-Host ""

# ============================================================
# RESUMEN FINAL
# ============================================================
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  EXPERIMENTOS COMPLETADOS" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Archivos CSV generados:" -ForegroundColor Green
Write-Host "  - results/seq/seq.csv" -ForegroundColor Gray
Write-Host "  - results/mpi/mpi_strong.csv" -ForegroundColor Gray
Write-Host "  - results/mpi/mpi_weak.csv" -ForegroundColor Gray
Write-Host "  - results/omp/omp_strong.csv" -ForegroundColor Gray
Write-Host "  - results/omp/omp_weak.csv" -ForegroundColor Gray
Write-Host "  - results/hybrid/hybrid_strong.csv" -ForegroundColor Gray
Write-Host "  - results/hybrid/hybrid_weak.csv" -ForegroundColor Gray
Write-Host ""
Write-Host "Siguiente paso: Genera las gráficas con:" -ForegroundColor Yellow
Write-Host "  python experiments/compare/compare_parallel_methods.py" -ForegroundColor Cyan
Write-Host ""
