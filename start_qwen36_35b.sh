#!/usr/bin/env bash
# Start Qwen3.6-35B-A3B NVFP4 via vLLM on GPU 1, port 9003 (model_manager proxies :8003→:9003)
# GPU layout: GPU 1 = this model alone (~18GB) / 32GB
set -euo pipefail

VENV_DIR="/home/derek/Projects/gemma4-bench/.venv"
LOG_DIR="/home/derek/Projects/llm-gateway/logs"
PID_FILE="/home/derek/Projects/llm-gateway/qwen36_35b.pid"
HF_HOME_DIR="/home/stardust/.cache/huggingface"

mkdir -p "$LOG_DIR"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Qwen3.6-35B-A3B already running with PID $(cat "$PID_FILE")"
  exit 0
fi

export HF_HOME="$HF_HOME_DIR"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-/home/stardust/.cache/huggingface/token}"
if [[ -z "${HF_TOKEN:-}" ]] && [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES=1
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"

MODEL_ID="${MODEL_ID:-RedHatAI/Qwen3.6-35B-A3B-NVFP4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-9003}"
API_KEY="${API_KEY:-local-qwen36}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.6-35b-a3b}"

echo "Starting Qwen3.6-35B-A3B NVFP4 on $HOST:$PORT (GPU 1) ..."

nohup "$VENV_DIR/bin/vllm" serve "$MODEL_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --api-key "$API_KEY" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --quantization compressed_tensors \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  >"$LOG_DIR/qwen36_35b.log" 2>&1 &

echo $! >"$PID_FILE"
echo "Started with PID $(cat "$PID_FILE") — logs: $LOG_DIR/qwen36_35b.log"
