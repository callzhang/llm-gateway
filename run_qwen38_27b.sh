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
# VISION ENABLED 2026-08-28 — --language-model-only dropped.  The old comment
# here claimed the flag saved no VRAM because qwen3_5.py builds self.visual
# unconditionally.  That was true of an older vLLM and is now WRONG: 0.25.1
# wraps the tower in `_mark_tower_model(vllm_config, {"image","video"})`
# (interfaces.py:250), and "tower model components are automatically skipped
# when --limit-mm-per-prompt is set to zero for all of their modalities".
# So the flag DID save the 0.86 GiB tower, and turning vision on costs it.
#
# MEASURED 2026-08-28, util 0.84 / max-model-len 65536 / free 32607 MiB card,
# two agreeing spawns per row (first-load trap does not apply — same
# checkpoint, already warm):
#   text-only (--language-model-only)     weights 16.74  KV 8.91 GiB  200,118 tok  3.05x
#   vision, uncapped image size           weights 17.60  KV 6.52 GiB  146,285 tok  2.23x
#   vision, longest_edge capped to 4 Mpx  weights 17.60  KV 7.93 GiB  177,883 tok  2.71x
#
# Uncapped costs 2.39 GiB of KV (-27% context budget) for only 0.86 GiB of
# tower.  The other ~1.5 GiB is multimodal memory profiling: preprocessor_config
# ships size.longest_edge = 16,777,216 px, which at 32x32 px/token (patch 16 x
# merge 2) is a 16,384-token image, and vLLM profiles one image at that maximum
# feature size.  Capping longest_edge to 4,194,304 px (4,096 tokens, ~2048x2048)
# recovers 1.41 GiB and leaves the total cost at 0.98 GiB / 22,235 tokens — i.e.
# essentially just the tower.  Raise the cap only against a re-measurement.
#
# --limit-mm-per-prompt video:0 keeps video frames out of the profile too.
# Floors re-measured with vision on and still valid: at the 0.78 util floor KV
# is 6.05 GiB / 135,753 tok / 2.07x — no collapse, so MODEL_MIN_GPU_MEM_UTIL
# and MODEL_MIN_FREE_GIB (25.6) stay as they are.
export HF_HOME=/home/stardust/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICE:-1}

# qwen3.8 MTP speculative decoding is disabled until vLLM/xgrammar
# accepted-prefix state synchronization is fixed upstream.
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONNOUSERSITE=1
VLLM_BIN=${VLLM_BIN:-/home/derek/miniforge3/envs/llm-gateway-vllm/bin/vllm}

HF_TOKEN_FILE=/home/stardust/.cache/huggingface/token
if [[ -f "$HF_TOKEN_FILE" ]]; then
  export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.88}   # model_manager lowers this to fit free VRAM.
                                          # 0.88 (2026-09-03, was 0.84) is the ceiling
                                          # that still leaves room for the lazy-loading
                                          # video-transcribe-service (~1.6 GiB on GPU1).
MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-131072}   # raised 65536→131072 2026-09-03 (native
                                             # 262144).  Changing this re-keys the
                                             # torch.compile cache and invalidates the
                                             # JIT warm marker — always let
                                             # llm-jit-warmup re-warm before serving.
                                             # Measured at 131072/0.88: KV 254,862 tok,
                                             # 1.94x full-length concurrency.
MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:?VLLM_MAX_NUM_SEQS is required}

# vLLM needs BOTH --tool-call-parser and --enable-auto-tool-choice: with
# the parser alone it rejects any request carrying tools.  Keep comments out
# of the exec's line continuation below — a '#' line there ends the command
# early.
exec "$VLLM_BIN" serve gittensor-model-hub/Qwen3.8-27B-NVFP4-RTX5090 \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9010} \
  --api-key local-qwen36 \
  --served-model-name qwen3.8-27b \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --max-model-len ${MAX_MODEL_LEN} \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":1,"video":0}' \
  --mm-processor-kwargs '{"size":{"longest_edge":4194304,"shortest_edge":65536}}' \
  --trust-remote-code
