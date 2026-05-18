#!/usr/bin/env bash
# Run Qwen3.6-35B-A3B NVFP4 via vLLM on GPU 0 — foreground for systemd (no nohup/PID file)
export HF_HOME=/home/stardust/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

HF_TOKEN_FILE=/home/stardust/.cache/huggingface/token
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

exec /home/derek/Projects/gemma4-bench/.venv/bin/vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --host 127.0.0.1 \
  --port 9002 \
  --api-key local-qwen36 \
  --served-model-name qwen3.6-35b-a3b \
  --quantization compressed-tensors \
  --gpu-memory-utilization 0.93 \
  --max-model-len 122880 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code
