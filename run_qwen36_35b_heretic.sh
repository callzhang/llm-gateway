#!/usr/bin/env bash
# Run Qwen3.6-35B-A3B heretic (uncensored, Heretic v1.2.0 abliteration) NVFP4 via vLLM.
# Weights from AEON-7; serving params mirror the proven run_qwen36_35b.sh baseline.
# Multimodal: the ViT vision tower (BF16) is bundled in the NVFP4 weights, so image
#   understanding works out of the box (no mmproj download — that's llama.cpp-only).
#   OCR proper is still served externally by rapidocr; this just enables image input.
# NOT included on purpose:
#   --speculative-config : needs a separate qwen36-dflash draft model + extra VRAM; run base first.
#   VLLM_TEST_FORCE_FP8_MARLIN / TORCH_MATMUL_PRECISION : DGX-Spark-specific from the card; 5090
#       has native NVFP4 kernels. Only add FP8_MARLIN=1 if startup errors on missing FP4 kernels.
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

GPU_MEM_UTIL=0.93
MAX_MODEL_LEN=122880

# served-model-name kept distinct so you can A/B against the stock 35b.
# To make this a drop-in replacement instead, rename to: qwen3.6-35b-a3b
exec /home/derek/Projects/gemma4-bench/.venv/bin/vllm serve AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9010} \
  --api-key local-qwen36 \
  --served-model-name qwen3.6-35b-a3b-heretic \
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
