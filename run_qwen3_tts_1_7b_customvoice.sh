#!/usr/bin/env bash
# Run Qwen3-TTS CustomVoice through the dedicated vLLM-Omni environment.
set -euo pipefail

export HF_HOME=${HF_HOME:-/home/stardust/.cache/huggingface}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICE:-0}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1

VLLM_BIN=${VLLM_TTS_BIN:-/home/derek/miniforge3/envs/llm-gateway-tts/bin/vllm-omni}
DEPLOY_CONFIG=/home/derek/Projects/llm-gateway/configs/qwen3_tts.yaml

HF_TOKEN_FILE=/home/stardust/.cache/huggingface/token
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(<"$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

exec "$VLLM_BIN" serve Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9000} \
  --api-key local-qwen36 \
  --served-model-name qwen3-tts-1.7b-customvoice \
  --deploy-config "$DEPLOY_CONFIG" \
  --trust-remote-code \
  --omni
