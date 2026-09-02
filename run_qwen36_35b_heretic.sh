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
export PYTHONNOUSERSITE=1
VLLM_BIN=${VLLM_BIN:-/home/derek/miniforge3/envs/llm-gateway-vllm/bin/vllm}

HF_TOKEN_FILE=/home/stardust/.cache/huggingface/token
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.93}   # model_manager lowers this to fit free VRAM
# Matched to stock 35b's 81920.  config.yaml's max_context_window_tokens must
# agree — trim_hook caps requests against it, so a config claiming more than
# vLLM loaded makes every over-length request fail at the backend.
# (Was temporarily 32768 while debugging the FlashInfer JIT cold-start hang,
# while config.yaml still claimed 122880 — that mismatch was live.)
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-81920}   # model_manager lowers this on a tight GPU to fit KV
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:?VLLM_MAX_NUM_SEQS is required}

# served-model-name kept distinct so you can A/B against the stock 35b.
# To make this a drop-in replacement instead, rename to: qwen3.6-35b-a3b
# vLLM needs BOTH --tool-call-parser and --enable-auto-tool-choice: with
# the parser alone it rejects any request carrying tools, with
# '"auto" tool choice requires --enable-auto-tool-choice and
# --tool-call-parser to be set'.  The second flag was missing from every
# run script since they were written, so tool calling had never worked
# through the gateway.  Keep comments out of the exec's line
# continuation below — a '#' line there ends the command early.
# max-num-seqs is a ceiling, not an allocation.  The scheduler already
# admits per request against actual free KV blocks
# (scheduler.py allocate_slots, with scheduler_reserve_full_isl=True), so
# long requests throttle concurrency down on their own.  It was 2, sized
# for every request filling max-model-len; real traffic has a ~6k-token
# median against 423846 tokens of KV, so that cap sat ~3% of capacity and
# the dynamic gate never got to act.
exec "$VLLM_BIN" serve AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4 \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9010} \
  --api-key local-qwen36 \
  --served-model-name qwen3.6-35b-a3b-heretic \
  --quantization compressed-tensors \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --trust-remote-code \
  --no-disable-hybrid-kv-cache-manager
