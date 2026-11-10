#!/bin/bash
# Run the whole S6E11 grid: 54 points, 24 parallel workers x 6 threads.
# One process per grid point, BLAS threads explicit, one log per worker,
# nohup-detached so an SSH disconnect cannot kill the run.
# Ascend toolkit, if present (public path, kept for reproducibility)
source "${ASCEND_TOOLKIT:-/usr/local/Ascend/ascend-toolkit}/set_env.sh" 2>/dev/null || true
PY=${S6E11_PY:-python3}
ROOT=${S6E11_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT" || exit 1

export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export MKL_NUM_THREADS=6

NPOINTS=$("$PY" code/s6e11_grid.py list 2>/dev/null | wc -l)
echo "grid points: $NPOINTS"
LAST=$((NPOINTS - 1))

run_one() {
  local IDX=$1
  "$PY" "$ROOT/code/s6e11_grid.py" "$IDX" 6 > "$ROOT/logs/point_${IDX}.log" 2>&1
  echo "$(date +%m%d_%H%M%S) point_${IDX} exit=$?" >> "$ROOT/logs/status.log"
}
export -f run_one
export PY ROOT

seq 0 "$LAST" | xargs -P 24 -I@ bash -c 'run_one @'
echo "GRID_COMPLETE $(date +%m%d_%H%M%S)"
touch "$ROOT/out/GRID_DONE.marker"
