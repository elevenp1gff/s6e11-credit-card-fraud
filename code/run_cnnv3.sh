#!/bin/bash
# Run the 5 folds of CNN v3 in parallel, one per NPU card (cards 1,2,3,7 are idle).
# Ascend toolkit, if present (public path, kept for reproducibility)
source "${ASCEND_TOOLKIT:-/usr/local/Ascend/ascend-toolkit}/set_env.sh" 2>/dev/null || true
PY=${S6E11_PY:-python3}
ROOT=${S6E11_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8

CARDS=(1 2 3 7 0)
run_fold() {
  local F=$1
  local CARD=${CARDS[$F]}
  ASCEND_RT_VISIBLE_DEVICES=$CARD "$PY" "$ROOT/code/cnnv3.py" "$F" \
    > "$ROOT/logs/cnnv3_f${F}.log" 2>&1
  echo "$(date +%m%d_%H%M%S) cnnv3_f${F} exit=$?" >> "$ROOT/logs/status.log"
}
export -f run_fold
export PY ROOT
declare -ax CARDS
export CARDS

for f in 0 1 2 3 4; do
  run_fold "$f" &
  sleep 5
done
wait
echo "CNNV3_COMPLETE $(date +%m%d_%H%M%S)"
touch "$ROOT/out/CNNV3_DONE.marker"
