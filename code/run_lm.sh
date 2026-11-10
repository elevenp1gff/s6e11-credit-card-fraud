#!/bin/bash
# Run the megayak target-first LM on Ascend NPU.
# Ascend toolkit, if present (public path, kept for reproducibility)
source "${ASCEND_TOOLKIT:-/usr/local/Ascend/ascend-toolkit}/set_env.sh" 2>/dev/null || true
PY=${S6E11_PY:-python3}
ROOT=${S6E11_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
cd "$ROOT" || exit 1
export OMP_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export MKL_NUM_THREADS=16
export ASCEND_RT_VISIBLE_DEVICES=4
export GPT2_LOCAL="$ROOT/gpt2_local"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export BATCH=${BATCH:-16}
export REP_BATCH=${REP_BATCH:-8}
export EPOCHS=${EPOCHS:-8}

MODE=${1:-check}
FOLDS=${2:-}
if [ -n "$FOLDS" ]; then
  "$PY" "$ROOT/code/lm_member_npu.py" "$MODE" --folds "$FOLDS" > "$ROOT/logs/lm_${MODE}_${FOLDS//,/_}.log" 2>&1
  echo "$(date +%m%d_%H%M%S) lm ${MODE} folds=${FOLDS} exit=$?" >> "$ROOT/logs/status.log"
else
  "$PY" "$ROOT/code/lm_member_npu.py" "$MODE" > "$ROOT/logs/lm_${MODE}.log" 2>&1
  echo "$(date +%m%d_%H%M%S) lm ${MODE} exit=$?" >> "$ROOT/logs/status.log"
fi
