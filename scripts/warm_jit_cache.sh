#!/usr/bin/env bash
# Pre-populate the FlashInfer JIT / autotune caches outside the request path.
#
# WHY THIS EXISTS
# ---------------
# These are NVFP4 checkpoints on sm_120 (RTX 5090).  FlashInfer ships no
# prebuilt fp4_gemm_cutlass_sm120, so the first vLLM start after a change to
# the model, FlashInfer, vLLM, or the driver compiles CUTLASS from source and
# then autotunes it.  Measured on 2026-07-20 for Qwen3.6-27B-NVFP4-MTP:
#
#   init engine (profile, create kv cache, warmup model) took 1606.72 s
#
# That is 2.7x model_manager's WAKE_TIMEOUT (600s), so on the request path the
# spawn is always killed mid-compile, the cache is never written, and the next
# request restarts the compile from zero.  That loop ran for 7 days with zero
# successful spawns, and each round drove the box to load 500+ (32 parallel
# nvcc/cicc peaking near 50 GiB RSS) which starved sshd/tailscaled/cloudflared
# and surfaced as Cloudflare 530/1033 on embed.preseen.ai.
#
# MAX_JOBS in gateway.env caps the compile swarm so it can no longer wedge the
# host.  This script covers the other half: it does the long first compile
# offline, with no timeout, so the request path only ever sees a warm cache.
# Once warm, cold start is ~126s — comfortably inside WAKE_TIMEOUT.
#
# Normally you do not run this by hand: llm-jit-warmup.service runs it as a
# oneshot before llm-model-manager.service, so :8002 stays closed (and LiteLLM
# fails fast) until every model is warm.  It is a no-op when the markers are
# already present, so it costs ~1s on a normal boot.
#
# WARMTH IS TRACKED BY MARKER FILE, NOT BY INSPECTING THE CACHE.  A compile that
# is killed partway leaves plenty of .o/.so files behind that look complete but
# are not — that is exactly how the 7-day loop stayed invisible.  A marker is
# written only after vLLM actually answered /health, and is keyed by a
# fingerprint of vllm + flashinfer + driver + GPU so any upgrade invalidates it.
#
# Usage:  scripts/warm_jit_cache.sh [--force] [run_script ...]
#         defaults to every run_qwen36_*.sh
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$(pwd)
LOGDIR=$ROOT/logs
mkdir -p "$LOGDIR"

set -a; . "$ROOT/gateway.env"; set +a
: "${MAX_JOBS:=4}"; export MAX_JOBS   # cap the nvcc swarm even if gateway.env lacks it

FORCE=0
SCRIPTS=()
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    *)       SCRIPTS+=("$arg") ;;
  esac
done
if [[ ${#SCRIPTS[@]} -eq 0 ]]; then
  mapfile -t SCRIPTS < <(ls "$ROOT"/run_qwen36_*.sh)
fi

PY=$ROOT/.venv/bin/python
pkgver() { "$PY" -c "import importlib.metadata as m; print(m.version('$1'))" 2>/dev/null || echo unknown; }

# Fingerprint everything whose change forces a recompile.  Bump any of these and
# the markers stop matching, so the next boot re-warms before serving traffic.
FP_RAW="vllm=$(pkgver vllm) flashinfer=$(pkgver flashinfer-python)"
FP_RAW+=" driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
FP_RAW+=" gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
FP=$(printf '%s' "$FP_RAW" | sha256sum | cut -c1-16)
MARKDIR=${XDG_CACHE_HOME:-$HOME/.cache}/llm-gateway/jit_warm/$FP
mkdir -p "$MARKDIR"

echo "fingerprint: $FP  ($FP_RAW)"
echo "markers:     $MARKDIR"

if systemctl --user is-active --quiet llm-model-manager.service; then
  echo "ERROR: llm-model-manager is running — it would fight this script for the GPU."
  echo "       systemctl --user stop llm-model-manager.service, then re-run."
  exit 1
fi

# Port 9000 = slot 0 = GPU 1, matching GPU_SLOTS in gateway.env.
export VLLM_CUDA_DEVICE=1 VLLM_PORT=9000

rc=0
for script in "${SCRIPTS[@]}"; do
  # Absolute path required: ionice execs via execvp, which resolves a bare name
  # through PATH and never through the cwd, so a relative arg fails with ENOENT.
  [[ $script == /* ]] || script=$ROOT/$script
  if [[ ! -f $script ]]; then
    echo "=== $(basename "$script") — no such run script, skipping ==="
    rc=1; continue
  fi
  name=$(basename "$script" .sh)
  # The run script's own contents are part of the key.  max_model_len changes
  # the torch.compile cache key (81920 hashes to a different cache dir than
  # 32768), and gpu-memory-utilization / quantization / backend flags matter
  # too — so editing a run script must invalidate its marker, or the next boot
  # would skip warmup and hand the recompile back to the request path.
  sha=$(sha256sum "$script" | cut -c1-12)
  marker=$MARKDIR/$name.$sha.ok
  # Drop markers for older revisions of this same script so they don't pile up.
  find "$MARKDIR" -maxdepth 1 -name "$name.*.ok" ! -name "$name.$sha.ok" -delete 2>/dev/null

  if [[ $FORCE -eq 0 && -f "$marker" ]]; then
    echo "=== $name — already warm ($(cat "$marker")) ==="
    continue
  fi

  # A warm run needs port 9000 to itself.  model_manager is stopped by now (we
  # checked), but it leaves its vLLM children running on purpose so the next
  # start can adopt them — so the port can still be held.  Say so plainly
  # instead of letting vLLM fail with a confusing bind error.
  if ss -ltn 2>/dev/null | grep -q ":$VLLM_PORT "; then
    echo "=== $name — port $VLLM_PORT still held by a running vLLM, skipping ==="
    echo "    model_manager leaves backends running for adoption; stop them first:"
    echo "      pkill -f 'vllm serve'"
    rc=1; continue
  fi

  log=$LOGDIR/warmup_$name.log
  echo "=== $name — compiling with MAX_JOBS=$MAX_JOBS (first run can exceed 30 min) ==="
  echo "    log: $log"

  # nice/ionice keep the compile behind sshd/tailscaled/cloudflared.  Children
  # inherit both, so the nvcc swarm is throttled too.
  setsid nice -n 10 ionice -c2 -n7 "$script" > "$log" 2>&1 &
  pgid=$!

  start=$(date +%s)
  ready=0
  # No timeout by design — being killed mid-compile is the bug we are fixing.
  while kill -0 "$pgid" 2>/dev/null; do
    if curl -sf --max-time 5 -o /dev/null "http://127.0.0.1:$VLLM_PORT/health"; then
      ready=1; break
    fi
    sleep 10
  done
  elapsed=$(( $(date +%s) - start ))

  if [[ $ready -eq 1 ]]; then
    echo "    OK — healthy in ${elapsed}s"
    # Marker is written only here: /health answered, so the cache is complete.
    echo "warmed $(date -Iseconds) in ${elapsed}s" > "$marker"
  else
    echo "    FAILED after ${elapsed}s — see $log"
    rc=1
  fi

  pkill -TERM -g "$pgid" 2>/dev/null
  for _ in $(seq 1 30); do pgrep -g "$pgid" >/dev/null || break; sleep 1; done
  pkill -KILL -g "$pgid" 2>/dev/null
  sleep 5
done

echo
echo "JIT cache now holds $(find ~/.cache/flashinfer -name '*.so' 2>/dev/null | wc -l) modules."
exit $rc
