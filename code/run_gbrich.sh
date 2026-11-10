#!/bin/bash
# Ascend toolkit, if present (public path, kept for reproducibility)
source "${ASCEND_TOOLKIT:-/usr/local/Ascend/ascend-toolkit}/set_env.sh" 2>/dev/null || true
PY=${S6E11_PY:-python3}
ROOT=${S6E11_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
export MKL_NUM_THREADS=32
"$PY" "$ROOT/code/gbrich.py" > "$ROOT/logs/gbrich.log" 2>&1
echo "$(date +%m%d_%H%M%S) gbrich exit=$?" >> "$ROOT/logs/status.log"
