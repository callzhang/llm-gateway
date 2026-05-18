#!/usr/bin/env bash
# Run Qwen3.6-27B NVFP4+MTP via vLLM — foreground for systemd (no nohup/PID file)
export HF_HOME=/home/stardust/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICE:-1}
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"

HF_TOKEN_FILE=/home/stardust/.cache/huggingface/token
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

exec /home/derek/Projects/gemma4-bench/.venv/bin/vllm serve sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9010} \
  --api-key local-qwen36 \
  --served-model-name qwen3.6-27b \
  --gpu-memory-utilization 0.84 \
  --max-model-len 65536 \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --language-model-only \
  --trust-remote-code
