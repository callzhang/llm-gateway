#!/usr/bin/env bash
# Start LLM Gateway:
#   OCR middleware  :8900  (OCR + reverse proxy to LiteLLM)
#   LiteLLM proxy   :8901  (model routing, fallbacks)
#   vLLM qwen3.6-27b   :9002 (GPU 0)
#   vLLM qwen3.6-35b-a3b :9003 (GPU 1, multimodal)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── Env defaults ──────────────────────────────────────────────────────────────
source "$SCRIPT_DIR/gateway.env" 2>/dev/null || true
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-local-gateway}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export LITELLM_BASE="${LITELLM_BASE:-http://127.0.0.1:8901}"

# ── Start vLLM backends (via model_manager on cold-start, not directly) ───────
echo "[gateway] Starting vLLM backends..."
nohup bash "$SCRIPT_DIR/run_qwen36_35b_gpu0.sh" > "$LOG_DIR/qwen36_35b_gpu0.log" 2>&1 &
nohup bash "$SCRIPT_DIR/run_qwen36_35b.sh"      > "$LOG_DIR/qwen36_35b.log"      2>&1 &

echo -n "[gateway] Waiting for qwen3.6-35b-a3b GPU0 (:9002)"
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:9002/health > /dev/null 2>&1; then echo " ready!"; break; fi
    echo -n "."; sleep 3
done

echo -n "[gateway] Waiting for qwen3.6-35b-a3b GPU1 (:9003)"
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:9003/health > /dev/null 2>&1; then echo " ready!"; break; fi
    echo -n "."; sleep 3
done

# ── Start LiteLLM ─────────────────────────────────────────────────────────────
echo "[gateway] Starting LiteLLM proxy on :8901 ..."
nohup "$VENV/bin/litellm" \
    --config "$SCRIPT_DIR/config.yaml" \
    --port 8901 \
    --host 127.0.0.1 \
    > "$LOG_DIR/litellm.log" 2>&1 &
echo "[gateway] LiteLLM PID=$!"

echo -n "[gateway] Waiting for LiteLLM"
for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8901/health -H "Authorization: Bearer $LITELLM_MASTER_KEY" > /dev/null 2>&1; then
        echo " ready!"; break
    fi
    echo -n "."; sleep 2
done

# ── Start gateway ─────────────────────────────────────────────────────────────
echo "[gateway] Starting LLM Gateway on :8900 ..."
nohup "$VENV/bin/python" "$SCRIPT_DIR/gateway.py" \
    > "$LOG_DIR/gateway.log" 2>&1 &
echo "[gateway] Gateway PID=$!"

sleep 2

echo ""
echo "Gateway ready:"
echo "  API     : http://127.0.0.1:8900/v1  (OPENAI_BASE_URL)"
echo "  OCR     : POST http://127.0.0.1:8900/v1/ocr"
echo "  LiteLLM : http://127.0.0.1:8901/ui"
echo "  Key     : $LITELLM_MASTER_KEY  (OPENAI_API_KEY)"
echo ""
echo "  Models  : qwen3.6-35b-a3b (GPU 0 + GPU 1, load-balanced)"
