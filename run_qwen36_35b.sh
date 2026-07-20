#!/usr/bin/env bash
# Run Qwen3.6-35B-A3B NVFP4 via vLLM — foreground for systemd (no nohup/PID file)
export HF_HOME=/home/stardust/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICE:-1}
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
VLLM_BIN=${VLLM_BIN:-/home/derek/Projects/llm-gateway/.venv/bin/vllm}

HF_TOKEN_FILE=/home/stardust/.cache/huggingface/token
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Both GPUs run identical config.  embedding-provider has idle-offload enabled
# (300s timeout) so when annotation traffic dominates, GPU 0 has full ~32 GiB
# available — same as the unshared GPU 1.  vLLM will refuse to start on GPU 0
# when embedding is actively loaded (~2.7 GiB) and util=0.93 doesn't fit; mm's
# scale-out cooldown handles the retry until embedding offloads.
GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.93}   # model_manager lowers this to fit free VRAM
# 81920, not the theoretical 122880: that was the largest context whose KV cache
# actually fits at util=0.93 on a 32 GiB card.  config.yaml's
# max_context_window_tokens must match — trim_hook caps requests against it.
# (Was temporarily 32768 while debugging the FlashInfer JIT cold-start hang.)
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-81920}   # model_manager lowers this on a tight GPU to fit KV

# vLLM needs BOTH --tool-call-parser and --enable-auto-tool-choice: with
# the parser alone it rejects any request carrying tools, with
# '"auto" tool choice requires --enable-auto-tool-choice and
# --tool-call-parser to be set'.  The second flag was missing from every
# run script since they were written, so tool calling had never worked
# through the gateway.  Keep comments out of the exec's line
# continuation below — a '#' line there ends the command early.
exec "$VLLM_BIN" serve RedHatAI/Qwen3.6-35B-A3B-NVFP4 \
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
  --enable-auto-tool-choice \
  --trust-remote-code \
  --no-disable-hybrid-kv-cache-manager
