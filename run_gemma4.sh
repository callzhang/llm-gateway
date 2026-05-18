#!/usr/bin/env bash
# Run Gemma4 E4B-it AWQ-INT4 via vLLM — foreground for systemd
# model_manager proxies :8001 → :9001
# GPU 0 shared with qwen3.6-27b; model_manager enforces only-one-awake policy.
export HF_HOME=/home/stardust/.cache/huggingface
export VLLM_SERVER_DEV_MODE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

HF_TOKEN_FILE=/home/stardust/.cache/huggingface/token
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

exec /home/derek/Projects/gemma4-bench/.venv/bin/vllm serve cyankiwi/gemma-4-E4B-it-AWQ-INT4 \
  --host 127.0.0.1 \
  --port 9001 \
  --api-key local-gemma4 \
  --served-model-name gemma4-e4b-it \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --gpu-memory-utilization 0.42 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --enable-sleep-mode
