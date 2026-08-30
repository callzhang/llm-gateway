#!/usr/bin/env bash
# Run Qwen3.8-27B NVFP4+MTP via vLLM — foreground for systemd (no nohup/PID file)
#
# CHECKPOINT CHOICE (measured 2026-08-27, safetensors headers, not repo metadata)
# ----------------------------------------------------------------------------
# Four NVFP4 quants of Qwen3.8-27B exist.  Loaded bytes and dtype composition:
#
#   gittensor…RTX5090  17.48 GiB   FP8 = weight_scale only  -> pure NVFP4
#   sakamakismile      19.15 GiB   FP8 = weight_scale only  -> pure NVFP4
#   RadixArk           20.42 GiB   6.72 GiB of FP8 *weights* (Mamba linear_attn)
#   unsloth            21.81 GiB   9.90 GiB of FP8 *weights* (+ lm_head)
#
# RadixArk/unsloth buy accuracy on quantisation-sensitive layers by leaving them
# at FP8 instead of FP4.  That is the right trade on a card with headroom; on a
# 32 GiB 5090 it is headroom we do not have, and the 4.33 GiB spread between
# gittensor and unsloth lands directly in the KV cache.  gittensor is the
# smallest of the four, so it is the pick.
#
# MEASURED (util 0.84, max-model-len 65536, 32607 MiB card).  Four readings —
# two warm_jit_cache runs and two production spawns via model_manager — all
# agree exactly:
#   qwen3.8-27b gittensor  weights 16.74 GiB  KV 8.91 GiB  200,118 tok  3.05x
#   qwen3.6-27b Text       weights 18.41 GiB  KV 7.25 GiB  162,669 tok  2.48x
# So 3.8 is lighter AND holds more KV: -1.67 GiB of weights turns into +1.66 GiB
# of cache, +37,449 tokens (+23%) of context budget.
#
# TRAP, cost us a wrong conclusion once: the FIRST spawn of a newly downloaded
# checkpoint under-reports KV badly — that run measured 6.65 GiB / 149,796 tok,
# 2.26 GiB below steady state, because first-load transients are still resident
# during vLLM's memory profiling.  Every later spawn returns 8.91 GiB.  Never
# calibrate a floor or compare checkpoints off a first-load reading; re-spawn
# once and use the second number.
#
# --language-model-only DOES NOT SAVE VRAM HERE.  qwen3_5.py builds
# self.visual unconditionally; the flag only makes get_limit_per_prompt()
# return 0, i.e. it rejects image/video *input*.  The 0.86 GiB BF16 vision
# tower loads either way.  It is kept for parity with the 3.6-27B slot this
# model shadows — dropping it unlocks vision at no extra weight cost, but
# changes multimodal profiling, so measure the floor again before doing so.
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

GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.84}   # model_manager lowers this to fit free VRAM
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-65536}   # 3.8 is native 262144; held at the 3.6-27B
                                             # value so KV demand per token is unchanged
                                             # and the JIT/torch.compile cache key is
                                             # comparable.  Raising it invalidates warmth.

# vLLM needs BOTH --tool-call-parser and --enable-auto-tool-choice: with
# the parser alone it rejects any request carrying tools.  Keep comments out
# of the exec's line continuation below — a '#' line there ends the command
# early.
# Do not enable MTP speculative decoding here.  In vLLM 0.25.1 it can desync
# xgrammar under concurrent JSON-schema output, allowing grammar-invalid tokens
# and causing otherwise valid structured responses to fail downstream parsing.
exec "$VLLM_BIN" serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9010} \
  --api-key local-qwen36 \
  --served-model-name qwen3.8-27b \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs 4 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --language-model-only \
  --trust-remote-code
