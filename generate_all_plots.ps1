# generate_all_plots.ps1
# Script de PowerShell para generar todas las gráficas de análisis
# 
# USO: 
#   .\generate_all_plots.ps1
#
# PREREQUISITOS: Haber ejecutado run_all_experiments.ps1 primero

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  Parallel KNN - Generar Todas las Gráficas" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Cambiar al directorio src/
Set-Location src

# ============================================================
# 1. Análisis Secuencial
# ============================================================
Write-Host "[1/8] Generando gráficas SECUENCIAL..." -ForegroundColor Green
python experiments/seq/analyze_seq.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en análisis secuencial" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/seq/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 2. Análisis MPI Strong
# ============================================================
Write-Host "[2/8] Generando gráficas MPI STRONG SCALING..." -ForegroundColor Green
python experiments/mpi/analyze_mpi_strong.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en análisis MPI strong" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/mpi/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 3. Análisis MPI Weak
# ============================================================
Write-Host "[3/8] Generando gráficas MPI WEAK SCALING..." -ForegroundColor Green
python experiments/mpi/analyze_mpi_weak.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en análisis MPI weak" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/mpi/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 4. Análisis OMP Strong
# ============================================================
Write-Host "[4/8] Generando gráficas OMP STRONG SCALING..." -ForegroundColor Green
python experiments/omp/analyze_omp_strong.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en análisis OMP strong" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/omp/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 5. Análisis OMP Weak
# ============================================================
Write-Host "[5/8] Generando gráficas OMP WEAK SCALING..." -ForegroundColor Green
python experiments/omp/analyze_omp_weak.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en análisis OMP weak" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/omp/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 6. Análisis Hybrid Strong
# ============================================================
Write-Host "[6/8] Generando gráficas HYBRID STRONG SCALING..." -ForegroundColor Green
python experiments/hybrid/analyze_hybrid_strong.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en análisis Hybrid strong" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/hybrid/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 7. Análisis Hybrid Weak
# ============================================================
Write-Host "[7/8] Generando gráficas HYBRID WEAK SCALING..." -ForegroundColor Green
python experiments/hybrid/analyze_hybrid_weak.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en análisis Hybrid weak" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/hybrid/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# 8. Comparación Global (MPI vs OMP vs Hybrid)
# ============================================================
Write-Host "[8/8] Generando gráficas de COMPARACIÓN GLOBAL..." -ForegroundColor Green
python experiments/compare/compare_parallel_methods.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR en comparación global" -ForegroundColor Red
    exit 1
}
Write-Host "    Completado: results/figures/compare/" -ForegroundColor Gray
Write-Host ""

# ============================================================
# RESUMEN FINAL
# ============================================================
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  GRÁFICAS GENERADAS EXITOSAMENTE" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Directorios con gráficas PNG:" -ForegroundColor Green
Write-Host "  - results/figures/seq/" -ForegroundColor Gray
Write-Host "  - results/figures/mpi/" -ForegroundColor Gray
Write-Host "  - results/figures/omp/" -ForegroundColor Gray
Write-Host "  - results/figures/hybrid/" -ForegroundColor Gray
Write-Host "  - results/figures/compare/  (comparación global)" -ForegroundColor Gray
Write-Host ""
Write-Host "Puedes abrir las gráficas con tu visor de imágenes preferido." -ForegroundColor Yellow
Write-Host ""
