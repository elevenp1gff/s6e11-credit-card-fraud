#!/bin/bash
# Run all 8 GLM-boost variants: 8 workers x 20 threads = 160 of 192 cores.
# Ascend toolkit, if present (public path, kept for reproducibility)
source "${ASCEND_TOOLKIT:-/usr/local/Ascend/ascend-toolkit}/set_env.sh" 2>/dev/null || true
PY=${S6E11_PY:-python3}
ROOT=${S6E11_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4

run_v() {
  local IDX=$1
  "$PY" "$ROOT/code/gbfam.py" "$IDX" > "$ROOT/logs/gbfam_${IDX}.log" 2>&1
  echo "$(date +%m%d_%H%M%S) gbfam_${IDX} exit=$?" >> "$ROOT/logs/status.log"
}
export -f run_v
export PY ROOT

for i in 0 1 2 3 4 5 6 7; do
  run_v "$i" &
done
wait
echo "GBFAM_COMPLETE $(date +%m%d_%H%M%S)"
touch "$ROOT/out/GBFAM_DONE.marker"
