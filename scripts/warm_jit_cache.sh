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
#         defaults to every run_qwen3[0-9]*_*.sh
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
  mapfile -t SCRIPTS < <(ls "$ROOT"/run_qwen3[0-9]*_*.sh)
fi

# The fingerprint must be computed with the interpreter that ACTUALLY compiles
# the kernels — the one the run scripts exec via VLLM_BIN — not with
# $ROOT/.venv, which no longer exists.  When it vanished, pkgver's
# "|| echo unknown" swallowed the ENOENT and every version silently became
# "unknown", so the fingerprint stopped tracking vllm/flashinfer entirely: an
# upgrade to either would no longer invalidate a single marker, which is the one
# job this fingerprint has.  It also silently re-keyed the whole marker
# directory, orphaning three warm models (2026-08-27).
VLLM_BIN=${VLLM_BIN:-/home/derek/miniforge3/envs/llm-gateway-vllm/bin/vllm}
PY=${WARM_PY:-$(dirname "$VLLM_BIN")/python}
if [[ ! -x $PY ]]; then
  echo "ERROR: no usable interpreter at $PY"
  echo "       The fingerprint would degrade to 'unknown' and stop invalidating"
  echo "       markers on upgrade.  Set WARM_PY (or VLLM_BIN) and re-run."
  exit 1
fi

# Prefer installed distribution metadata; fall back to the module's __version__
# for packages whose dist-info is not resolvable from this interpreter.
pkgver() {
  "$PY" - "$1" <<'PYEOF' 2>/dev/null || echo unknown
import importlib, importlib.metadata as md, sys
name = sys.argv[1]
try:
    print(md.version(name)); raise SystemExit
except SystemExit:
    raise
except Exception:
    pass
mod = {"flashinfer-python": "flashinfer"}.get(name, name)
try:
    print(getattr(importlib.import_module(mod), "__version__", "unknown"))
except Exception:
    print("unknown")
PYEOF
}

# Fingerprint everything whose change forces a recompile.  Bump any of these and
# the markers stop matching, so the next boot re-warms before serving traffic.
FP_RAW="vllm=$(pkgver vllm) flashinfer=$(pkgver flashinfer-python)"
FP_RAW+=" driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"
FP_RAW+=" gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
# Refuse to run on a degraded fingerprint.  Warming under "unknown" writes
# markers that survive an upgrade they should not survive, and re-keys the
# marker directory so every already-warm model looks cold.  Fail loudly instead.
if [[ $FP_RAW == *unknown* ]]; then
  echo "ERROR: could not resolve a version for the fingerprint: $FP_RAW"
  echo "       Interpreter used: $PY"
  echo "       Refusing to warm under a fingerprint that cannot detect upgrades."
  exit 1
fi
FP=$(printf '%s' "$FP_RAW" | sha256sum | cut -c1-16)
MARKDIR=${XDG_CACHE_HOME:-$HOME/.cache}/llm-gateway/jit_warm/$FP
mkdir -p "$MARKDIR"

echo "fingerprint: $FP  ($FP_RAW)"
echo "markers:     $MARKDIR"

# The markers live outside the caches they vouch for, so wiping a cache without
# wiping the markers would leave us claiming "already warm" while the compile is
# actually gone — and that compile would then land on the request path, where
# WAKE_TIMEOUT still applies.  Cross-check both cache roots and drop every
# marker if either has been cleared.
# Test for compiled artefacts, not for the directories: a running vLLM recreates
# an empty ~/.cache/vllm the moment it is removed, so "the directory exists" and
# even "it has an entry" both pass against a cache that is actually gone.
CACHE_DIR=${XDG_CACHE_HOME:-$HOME/.cache}
has_artefact() {  # <dir> <find-args...>
  local d=$1; shift
  [ -n "$(find "$d" "$@" -print -quit 2>/dev/null)" ]
}
if ! has_artefact "$CACHE_DIR/flashinfer" -type f -name '*.so' \
   || ! has_artefact "$CACHE_DIR/vllm/torch_compile_cache" -type f; then
  echo "  compiled artefacts missing under $CACHE_DIR — discarding markers, re-warming all"
  rm -f "$MARKDIR"/*.ok 2>/dev/null
fi

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
  # model_manager injects VLLM_MAX_NUM_SEQS at spawn; a warm run invokes the
  # run script directly, so it must supply the same value itself or the
  # script's ":?required" guard kills the warm attempt in seconds (which is
  # exactly what a marker-invalidated re-warm hit on 2026-09-03).  Keep these
  # in sync with MODEL_CONFIGS in model_manager.py.
  case $name in
    run_qwen38_27b)         export VLLM_MAX_NUM_SEQS=8 ;;
    run_qwen36_35b_heretic) export VLLM_MAX_NUM_SEQS=16 ;;
    *)                      unset VLLM_MAX_NUM_SEQS ;;
  esac
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
