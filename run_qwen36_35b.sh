#!/usr/bin/env bash
# Run Qwen3.6-35B-A3B NVFP4 via vLLM — foreground for systemd (no nohup/PID file)
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

# GPU 0 shares with embedding-provider (~2.2 GiB idle, up to ~2.73 GiB under load).
# Must satisfy two constraints:
#   (1) startup check: util × 31.36 GiB < free_at_scaleout
#       embedding peaks at ~2.73 GiB → free = 28.63 GiB
#       0.912 × 31.36 = 28.60 GiB < 28.63 GiB → 30 MiB margin ✓
#       (was 0.914 → 28.66 GiB > 28.63 GiB → failed during high-load scale-out)
#   (2) KV cache min:  at 0.912 util, available KV ≈ 0.996 GiB (~42 blocks × 2096-tok)
#       max_model_len must need ≤ 42 blocks; 81920 needs 40 blocks ✓
#       122880 would need 59 blocks — too large for GPU 0.
# GPU 1 is unshared; keep 0.93 util and full 122880 context.
if [[ "${VLLM_CUDA_DEVICE:-1}" == "0" ]]; then
  GPU_MEM_UTIL=0.912
  MAX_MODEL_LEN=81920   # GPU 0: reduced context (KV cache constraint); 40 blocks × 2096
else
  GPU_MEM_UTIL=0.93
  MAX_MODEL_LEN=122880  # GPU 1: unshared, full context window
fi

exec /home/derek/Projects/gemma4-bench/.venv/bin/vllm serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9010} \
  --api-key local-qwen36 \
  --served-model-name qwen3.6-35b-a3b \
  --quantization compressed-tensors \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs 2 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --trust-remote-code \
  --no-disable-hybrid-kv-cache-manager
