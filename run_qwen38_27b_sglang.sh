#!/usr/bin/env bash
# Native process: the existing manager owns the session/process group and lifecycle.
set -euo pipefail
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES="${VLLM_CUDA_DEVICE:?manager GPU required}"
export CUDA_HOME=/data/derek/venv-vllm029/lib/python3.12/site-packages/nvidia/cu13
export LD_LIBRARY_PATH="/data/derek/sglang-runtime/cuda-lib:${LD_LIBRARY_PATH:-}"
export PATH="/data/derek/venv-sglang0517/bin:$CUDA_HOME/bin:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/data/derek/sglang-runtime/cache/triton
export TORCHINDUCTOR_CACHE_DIR=/data/derek/sglang-runtime/cache/torchinductor
export FLASHINFER_WORKSPACE_BASE=/data/derek/sglang-runtime/cache
export MAX_JOBS=8
export SGLANG_PLUGINS=friday_tool_json
# friday_thinking_budget: the qwen3.8 thinking-budget processor model_manager names for requests that carry
# thinking_token_budget (SGLang ignores that vLLM option); loaded by reference, so it must be importable here.
export PYTHONPATH="$(cd "$(dirname "$0")" && pwd)/sglang_plugins${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_IMAGE_MAX_PIXELS=4194304 SGLANG_IMAGE_MIN_PIXELS=65536
# --constrained-json-disable-any-whitespace: xgrammar otherwise allows unlimited whitespace between JSON tokens.
# When the model quoted a phrase with a bare `"` inside a string value, the string ended early and the grammar
# left it nothing but whitespace, so it padded to the output ceiling (CN memory's judges: 16,000 tokens, ~575 s
# each, up to 25 an hour on 2026-11-17).
# SGLang takes the key from the command line or a --config file only.  The file is rewritten at each
# spawn; mv keeps a concurrent spawn from reading a half-written (keyless) one.
: "${ENGINE_API_KEY:?set ENGINE_API_KEY in gateway.env}"
key_config="$(cd "$(dirname "$0")" && pwd)/engine-api-key.yaml"
key_tmp=$(mktemp "$key_config.XXXXXX")   # mode 600
printf "api-key: '%s'\n" "$ENGINE_API_KEY" > "$key_tmp"
mv -f "$key_tmp" "$key_config"
exec /data/derek/venv-sglang0517/bin/python -m sglang.launch_server \
 --model-path /data/zhoujunjie/models/Qwen3.8-27B-NVFP4 \
 --served-model-name qwen3.8-27b --host 127.0.0.1 --port "${VLLM_PORT:?manager port required}" \
 --tp-size 1 --quantization modelopt_fp4 --fp4-gemm-backend marlin --dtype bfloat16 \
 --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer \
 --limit-mm-data-per-request '{"image":1,"video":0,"audio":0}' \
 --mm-feature-transport cpu --mamba-radix-cache-strategy extra_buffer_lazy \
 --mamba-full-memory-ratio 2 --linear-attn-backend triton --mamba-backend triton \
 --mem-fraction-static "${VLLM_GPU_MEM_UTIL:?manager memory budget required}" \
 --context-length "${VLLM_MAX_MODEL_LEN:-131072}" --chunked-prefill-size 4096 \
 --max-running-requests "${VLLM_MAX_NUM_SEQS:?manager capacity required}" \
 --max-queued-requests 128 --cuda-graph-max-bs-decode 32 \
 --grammar-backend xgrammar --constrained-json-disable-any-whitespace \
 --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
 --enable-custom-logit-processor --enable-cache-report --enable-metrics --config "$key_config"
