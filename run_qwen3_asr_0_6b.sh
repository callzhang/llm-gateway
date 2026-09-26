#!/usr/bin/env bash
# Run Qwen3-ASR-0.6B (speech-to-text) with vLLM online FP8 quantization.
# Co-resident model: model_manager gives it a VRAM budget on whichever GPU has
# room, instead of a slot.  Serves POST /v1/audio/transcriptions.
set -euo pipefail

# Weights (and the ForcedAligner used for timestamps) are cached here; never let
# a cold start block on the Hub.
export HF_HOME=${HF_HOME:-/home/derek/services/asr-provider/runtime-cache/qwen3-asr}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICE:-0}
export PYTHONNOUSERSITE=1
export VLLM_NO_USAGE_STATS=1

VLLM_BIN=${VLLM_ASR_BIN:-/home/derek/miniforge3/envs/llm-gateway-vllm-029/bin/vllm}

# --quantization fp8: weights are converted from the official BF16 checkpoint at
# load (no calibration, dynamic activation scales).  gpu-memory-utilization
# comes from model_manager (VLLM_GPU_MEM_UTIL): preferred 0.10, lowered to fit the
# GPU's current free VRAM, floor 0.08.  --enforce-eager skips CUDA-graph capture,
# which costs memory a shared GPU does not have and buys nothing for <=60 s clips.
exec "$VLLM_BIN" serve Qwen/Qwen3-ASR-0.6B \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9100} \
  --api-key local-qwen36 \
  --served-model-name qwen3-asr-0.6b \
  --quantization fp8 \
  --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL:-0.10} \
  --max-model-len ${VLLM_MAX_MODEL_LEN:-2048} \
  --max-num-seqs ${VLLM_MAX_NUM_SEQS:-4} \
  --enforce-eager
