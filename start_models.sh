#!/usr/bin/env bash
# Start all vLLM backends in the correct order.
# GPU 0 conflict: gemma4 (AWQ, ~12 GB) + qwen3.6-27b (~20 GB) > 32 GB.
# Solution: start 27B → wait healthy → sleep it → start gemma4.
# GPU 1 (35B, ~27 GB) is independent and starts in parallel.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

wait_health() {
    local url="$1" label="$2" secs="${3:-180}"
    echo -n "[models] Waiting for $label"
    for i in $(seq 1 $((secs/3))); do
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

# ── GPU 1: start 35B immediately (independent) ───────────────────────────────
echo "[models] Starting qwen3.6-35b-a3b on GPU 1 (:9003) ..."
nohup bash "$SCRIPT_DIR/run_qwen36_35b.sh" > "$LOG_DIR/qwen36_35b.log" 2>&1 &
echo "[models] 35B PID=$!"

# ── GPU 0 step 1: start 27B ───────────────────────────────────────────────────
echo "[models] Starting qwen3.6-27b on GPU 0 (:9002) ..."
nohup bash "$SCRIPT_DIR/run_qwen36_27b.sh" > "$LOG_DIR/qwen36_27b.log" 2>&1 &
wait_health "http://127.0.0.1:9002/health" "qwen3.6-27b" 300

# ── GPU 0 step 2: sleep 27B so gemma4 can load ───────────────────────────────
echo "[models] Sleeping 27B to free GPU 0 for gemma4 ..."
curl -sf -X POST "http://127.0.0.1:9002/sleep?level=1" > /dev/null 2>&1 || true
sleep 10  # allow offload to finish

# ── GPU 0 step 3: start gemma4 ───────────────────────────────────────────────
echo "[models] Starting gemma4-e4b-it on GPU 0 (:9001) ..."
nohup bash "$SCRIPT_DIR/run_gemma4.sh" > "$LOG_DIR/gemma4.log" 2>&1 &
wait_health "http://127.0.0.1:9001/health" "gemma4-e4b-it" 300

echo "[models] All vLLM backends started."
