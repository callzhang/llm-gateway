#!/usr/bin/env bash
# Drive a vLLM backend directly, for debugging with model_manager stopped.
#
# WHAT THIS USED TO BE
# --------------------
# It sequenced qwen3.6-27b and gemma4 onto GPU 0 with a sleep/wake dance
# because the two could not fit there together.  Every part of that premise is
# gone: gemma4 is not a registered service, qwen3.6-27b was retired 2026-08-27
# (weights deleted, run script removed), and :9001/:9002/:9003 are not the slot
# ports.  The script had been dead for months and silently referenced files that
# no longer exist.
#
# WHAT IT IS NOW
# --------------
# model_manager owns the GPUs — it spawns on demand, keeps one model per slot,
# and idle-unloads.  This script is only for driving a backend by hand while
# model_manager is stopped.  That is the same rule scripts/warm_jit_cache.sh
# enforces, for the same reason: two owners racing for one card produces an OOM
# that reads like a model bug.
#
# There are two slots (GPU 0 -> :9000, GPU 1 -> :9010, per GPU_PORT_BASE and
# GPU_PORT_GAP) and more registered models than slots, so "start everything" is
# not a thing that can be asked for.  Name at most two.
#
# Usage:  ./start_models.sh [model ...]
#         default: qwen3.8-27b qwen3.6-35b-a3b-heretic
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# Must track MODEL_CONFIGS in model_manager.py.  Kept as an explicit table
# rather than parsed out of the Python so a typo fails here, loudly, instead of
# launching the wrong checkpoint.
declare -A RUN_SCRIPT=(
  ["qwen3.8-27b"]="run_qwen38_27b.sh"
  ["qwen3.6-35b-a3b-heretic"]="run_qwen36_35b_heretic.sh"
)

# slot index -> GPU id and port, matching _discover_gpu_slots()'s auto layout.
SLOT_GPU=(0 1)
SLOT_PORT=(9000 9010)
MAX_SLOTS=${#SLOT_PORT[@]}

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=("qwen3.8-27b" "qwen3.6-35b-a3b-heretic")
fi

if [[ ${#MODELS[@]} -gt $MAX_SLOTS ]]; then
  echo "[models] ERROR: asked for ${#MODELS[@]} models but only $MAX_SLOTS slots exist."
  echo "         Registered models outnumber slots by design; name at most $MAX_SLOTS."
  exit 1
fi

for m in "${MODELS[@]}"; do
  if [[ -z ${RUN_SCRIPT[$m]:-} ]]; then
    echo "[models] ERROR: unknown model '$m'. Known: ${!RUN_SCRIPT[*]}"
    exit 1
  fi
  if [[ ! -x $SCRIPT_DIR/${RUN_SCRIPT[$m]} ]]; then
    echo "[models] ERROR: ${RUN_SCRIPT[$m]} missing or not executable."
    exit 1
  fi
done

if systemctl --user is-active --quiet llm-model-manager.service; then
  echo "[models] ERROR: llm-model-manager is running — it owns the GPUs and would"
  echo "         fight this script for them."
  echo "         systemctl --user stop llm-model-manager.service, then re-run."
  exit 1
fi

wait_health() {
    local url="$1" label="$2" secs="${3:-300}"
    echo -n "[models] Waiting for $label"
    for _ in $(seq 1 $((secs / 3))); do
        if curl -sf "$url" > /dev/null 2>&1; then
            echo " ready"
            return 0
        fi
        echo -n "."
        sleep 3
    done
    echo " TIMEOUT"
    return 1
}

rc=0
LAUNCHED_MODEL=()
LAUNCHED_PORT=()
for i in "${!MODELS[@]}"; do
  m=${MODELS[$i]}
  gpu=${SLOT_GPU[$i]}
  port=${SLOT_PORT[$i]}

  if ss -ltn 2>/dev/null | grep -q ":$port "; then
    echo "[models] :$port already in use — skipping $m."
    echo "         A previous backend may still be shutting down; model_manager"
    echo "         leaves children running on purpose so a restart can adopt them."
    rc=1
    continue
  fi

  echo "[models] Starting $m on GPU $gpu (:$port) ..."
  VLLM_CUDA_DEVICE=$gpu VLLM_PORT=$port \
    nohup bash "$SCRIPT_DIR/${RUN_SCRIPT[$m]}" > "$LOG_DIR/manual_${m//./_}.log" 2>&1 &
  echo "[models] $m PID=$!  log: $LOG_DIR/manual_${m//./_}.log"
  LAUNCHED_MODEL+=("$m")
  LAUNCHED_PORT+=("$port")
done

# Health-check only what we actually launched, and only after launching them
# all, so a pair of ~50 s cold starts overlaps instead of serialising.  Testing
# the port here instead would skip everything: nothing has bound yet.
for i in "${!LAUNCHED_MODEL[@]}"; do
  wait_health "http://127.0.0.1:${LAUNCHED_PORT[$i]}/health" "${LAUNCHED_MODEL[$i]}" 300 || rc=1
done

if [[ $rc -eq 0 ]]; then
  echo "[models] Started: ${MODELS[*]}"
  echo "[models] Remember to stop these before restarting llm-model-manager."
else
  echo "[models] Finished with problems — see logs above."
fi
exit $rc
