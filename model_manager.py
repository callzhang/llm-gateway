#!/usr/bin/env python3
"""
model_manager.py — Dynamic GPU allocation proxy for vLLM backends.

GPU slots are a shared pool.  At most one model runs per slot at a time.

Routing rules:
  1. Model already running on ≥1 slots  → round-robin across those instances.
  2. Model is mid-spawn on a slot       → wait for it (don't double-spawn).
  3. Model not running, free slot exists → claim slot, spawn, serve.
  4. No free slot (all occupied by other models) → 503 gpu_busy immediately.

Scale-out:
  When all running instances of a model have _active_requests > 0 and a free
  slot exists, a second instance is spawned in the background.  Round-robin
  covers both once it is ready.

Idle unload:
  IDLE_TIMEOUT seconds with no requests → kill instance, release slot for reuse.

Scripts receive VLLM_CUDA_DEVICE and VLLM_PORT env vars at launch so the same
script can run on any slot.  Scripts must honour these variables.

Environment overrides:
  IDLE_TIMEOUT   idle seconds before unload   (default: 1200)
  WAKE_TIMEOUT   max seconds for cold start   (default: 300)
  HEALTH_POLL    poll interval while waking   (default: 2.0)
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import signal
import subprocess
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Literal

import aiohttp
from aiohttp import web

# ── Configuration ──────────────────────────────────────────────────────────────
# 300→1200 (2026-09-04): real traffic for both chat models arrives in bursts
# 10-90 min apart, so 300s kept unloading models whose next burst almost always
# came back — a dozen+ pointless unload→cold-start (~110s) cycles a day,
# including an unload beaten by the next request within 2 minutes (11:44→11:46).
# 20 min keeps the daytime cadence warm while still releasing the cards across
# the long overnight gaps.  A service-wide knob on purpose: idle policy is
# resource management, not a property of any one model.  Replicas still shed
# early via REPLICA_IDLE_TIMEOUT.
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "1200"))
# Not sized for the normal case — warm-cache spawns finish in 40-160s.  This is
# the self-heal budget for when llm-jit-warmup did not run or its marker was
# wrong, so a first-time FlashInfer/CUTLASS compile lands on the request path
# instead.  Those take 750-1890s; anything shorter kills the spawn before the
# JIT cache is written, so every retry restarts from zero and the model never
# comes up.  gateway.env normally sets this, but that file is gitignored —
# defaulting here keeps the guard in tracked code.
WAKE_TIMEOUT = int(os.environ.get("WAKE_TIMEOUT", "2700"))
HEALTH_POLL  = float(os.environ.get("HEALTH_POLL", "2.0"))
LISTEN_PORT  = int(os.environ.get("LISTEN_PORT", "8002"))
# Bind 0.0.0.0 so tailnet peers can reach it (Tailscale's ts-input iptables
# chain ACCEPTs tailscale0 traffic before ufw; LAN stays blocked by ufw
# default-deny; 0.0.0.0 still covers 127.0.0.1 so internal loopback is intact).
LISTEN_HOST  = os.environ.get("LISTEN_HOST", "0.0.0.0")

# ── Scale-out gating (weighted backlog accumulator) ──────────────────────────────
# Scale a model onto a 2nd GPU based on its *real* internal queue
# (num_requests_waiting, summed across instances).  waiting > 0 already means vLLM
# can't admit the request into the current batch (max_num_seqs reached or KV full)
# — the instance is AT CAPACITY — so the depth of the backlog sets the urgency.
#
# SCALE_OUT_TIERS maps a sustained waiting depth → seconds-at-that-depth to fire.
# A per-model progress P accumulates `dt / sustain(waiting)` each sample and fires
# at P ≥ 1.0.  Deeper backlog accrues faster, and time at a shallower depth counts
# proportionally toward a deeper threshold (e.g. waiting=2 accrues 1/300 per sec =
# 0.66× the 1/200 of waiting=3).  Edges:
#   • waiting == 0  → queue cleared, reset P to 0.
#   • waiting == 1  → below the lowest tier; HOLD P (no accrual, no reset).
#   • waiting >= 2  → accrue at the matching tier's rate.
# P is in-memory only (not persisted) — a model_manager restart resets it.
# Format: "depth:seconds,..."  (default: 4→100s, 3→200s, 2→300s).
def _parse_scale_tiers(spec: str) -> "list[tuple[int, float]]":
    tiers = []
    for part in spec.split(","):
        depth, secs = part.split(":")
        tiers.append((int(depth), float(secs)))
    return sorted(tiers, key=lambda t: -t[0])   # deepest first

SCALE_OUT_TIERS = _parse_scale_tiers(
    os.environ.get("SCALE_OUT_TIERS", "4:100,3:200,2:300")
)
SCALE_OUT_MIN_DEPTH = SCALE_OUT_TIERS[-1][0]   # lowest tier depth (default 2)

# Sliding window for the accumulator: P is the sum of per-sample accruals from
# the last SCALE_WINDOW seconds; older samples fall out on their own.  This
# replaces the old waiting==0 hard reset — an oscillating backlog (queue
# repeatedly forming and draining while the batch stays pinned at max_num_seqs)
# now accumulates across episodes instead of starting over each time, while
# evidence older than the window can never contribute to a fire.  Must exceed
# the largest tier sustain (default 300s) or that tier can never fire; 2-3×
# is the useful range.
SCALE_WINDOW = float(os.environ.get("SCALE_WINDOW", "900"))

def _scale_sustain_for(waiting: int) -> "float | None":
    """Seconds-at-this-depth needed to fire, for the deepest tier ≤ waiting.
    None if waiting is below the lowest tier (no accrual)."""
    for depth, secs in SCALE_OUT_TIERS:        # descending
        if waiting >= depth:
            return secs
    return None
# Extra replicas (a model running on >1 slot) idle out faster than the primary so
# a borrowed slot is returned to its evicted model promptly (asymmetric scale-in).
REPLICA_IDLE_TIMEOUT = int(os.environ.get("REPLICA_IDLE_TIMEOUT", "120"))
# A replica also has to see the model QUIET — no queued request on any instance
# — for this long before it is reclaimed.  Its own idleness is not enough:
# least-connections gives the replica the lighter share, so it can fall idle
# while the primary is still saturated, and the instantaneous "is anything
# queued right now" check passes during any brief trough.  Observed 2026-09-04:
# replica reclaimed at 23:26:54 during a trough, backlog returned 50s later,
# scale-out re-fired at 23:29:14 — a wasted ~110s cold start with 5-6 requests
# queued behind it.  Must exceed the scale-out tiers' reaction time.
# Calibrated against measured gap distributions, not picked.  Over three days of
# request arrivals, bucketed per model by which one was serving at the time:
#
#   qwen3.8-27b   6040 reqs, 136 gaps >60s: p50=118s p75=252s p90=783s
#     coverage 300s→76.5%  600s→87.5%  900s→92.6%  1200s→93.4%
#     GPU held  160 min     170 min     150 min     180 min
#   heretic       1976 reqs,  14 gaps >60s: p50=133s p75=591s p90=941s
#     coverage 300s→71.4%  600s→78.6%  900s→85.7%  1200s→85.7%
#
# Held-GPU cost is (gaps ≥ threshold) × threshold, and it does NOT rise
# monotonically: raising the bar past a cluster of within-burst pauses removes
# more gaps than the longer hold adds, so 900s actually holds *less* GPU than
# 600s while covering 5 points more.  1200s is past the knee (+0.8 points,
# +30 min).  Both models agree on 900s, which is why this stays a service-wide
# knob rather than a per-model table.
#
# Measure from request ARRIVALS (aiohttp access log), never from a vLLM
# instance's own log: an instance is unloaded after IDLE_TIMEOUT, so its log
# cannot contain a gap longer than that and the tail gets silently censored —
# which made an earlier attempt at this read "1200s covers 100%".
REPLICA_QUIET_BEFORE_RECLAIM = int(
    os.environ.get("REPLICA_QUIET_BEFORE_RECLAIM", "900")
)

# ── Self-heal: recycle a ready-but-degraded backend on repeated upstream 5xx ──────
# A backend can be process-alive yet broken — CUDA error, wedged scheduler, or an
# EngineCore that crashed but left the API server returning 500s.  The idle/crash
# watchdog only checks process liveness, so it won't catch this.  When a *ready*
# backend returns RECYCLE_5XX_THRESHOLD consecutive upstream 5xx, kill it and free
# the slot so the next request spawns a fresh instance.  EngineCore crashes recycle
# on the first hit (known fatal).  RECYCLE_COOLDOWN (slot-level, survives respawn)
# rate-limits recycle→respawn so a request-driven 500 loop can't thrash the GPU
# (each respawn costs a ~90s cold start).
RECYCLE_5XX_THRESHOLD = int(os.environ.get("RECYCLE_5XX_THRESHOLD", "3"))
RECYCLE_COOLDOWN      = int(os.environ.get("RECYCLE_COOLDOWN", "180"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR    = os.path.join(SCRIPT_DIR, "logs")

# ── GPU slot discovery ─────────────────────────────────────────────────────────
# Slots are the physical GPU resources available to this process.
# Each slot maps (slot_id, gpu_id, api_port).  model_manager sets
# VLLM_CUDA_DEVICE=gpu_id and VLLM_PORT=api_port when spawning a backend.
#
# Port spacing MUST be > 2 per slot.  vLLM's EngineCore subprocess binds
# api_port + N (typically +2) for its internal ZMQ IPC socket.  If the next
# slot's api_port falls within that range, spawns will fail with
# "Address already in use".  The default gap of 10 is conservative and safe.
#
# Configuration (env vars, evaluated in order):
#
#   GPU_SLOTS="0:9000,1:9010,2:9020"
#       Explicit gpu_id:port pairs, comma-separated.  Use this to skip GPUs
#       that are reserved for other workloads (e.g. GPU 0 running a desktop).
#
#   GPU_IDS="0,2,4"
#       Restrict auto-detection to these GPU indices.
#
#   GPU_PORT_BASE=9000   (default: 9000)
#   GPU_PORT_GAP=10      (default: 10)
#       Auto-mode: first slot gets GPU_PORT_BASE, next gets +GPU_PORT_GAP, etc.

_PORT_BASE = int(os.environ.get("GPU_PORT_BASE", "9000"))
_PORT_GAP  = int(os.environ.get("GPU_PORT_GAP",  "10"))


def _discover_gpu_slots() -> list[tuple[int, int, int]]:
    """Return [(slot_id, gpu_id, api_port), ...] from env or nvidia-smi."""

    # 1. Fully explicit override
    if slot_str := os.environ.get("GPU_SLOTS", "").strip():
        slots = []
        for i, token in enumerate(slot_str.split(",")):
            token = token.strip()
            if ":" in token:
                gpu_s, port_s = token.split(":", 1)
                slots.append((i, int(gpu_s), int(port_s)))
            else:
                # Just a GPU id — auto-assign port
                slots.append((i, int(token), _PORT_BASE + i * _PORT_GAP))
        return slots

    # 2. Auto-detect via nvidia-smi
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        all_gpu_ids = [int(x) for x in out.splitlines() if x.strip().isdigit()]
    except Exception:
        all_gpu_ids = []

    if not all_gpu_ids:
        # nvidia-smi failed, timed out, or returned nothing (e.g. drivers still
        # initialising at boot).  Fall back to GPU 0 so the service can start.
        logging.getLogger("model_manager").warning(
            "nvidia-smi returned no GPU indices — falling back to GPU 0. "
            "Set GPU_SLOTS env var for explicit configuration."
        )
        all_gpu_ids = [0]

    # 3. Optional filter
    if ids_str := os.environ.get("GPU_IDS", "").strip():
        wanted = {int(x) for x in ids_str.split(",") if x.strip().isdigit()}
        all_gpu_ids = [g for g in all_gpu_ids if g in wanted]

    return [
        (i, gpu_id, _PORT_BASE + i * _PORT_GAP)
        for i, gpu_id in enumerate(all_gpu_ids)
    ]


GPU_SLOTS: list[tuple[int, int, int]] = _discover_gpu_slots()


def _gpu_free_mib(gpu_id: int) -> float | None:
    """Return free GPU memory in MiB for the given GPU, or None on error."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return float(out.splitlines()[0]) if out else None
    except Exception:
        return None


def _gpu_total_mib(gpu_id: int) -> float | None:
    """Return total GPU memory in MiB for the given GPU, or None on error."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return float(out.splitlines()[0]) if out else None
    except Exception:
        return None


def _gpu_vllm_used_mib(gpu_id: int) -> float:
    """Return total GPU memory (MiB) used by all vLLM processes on the given GPU.
    Includes EngineCore subprocesses which hold most of the VRAM.  Returns 0 on
    error so callers can still do a conservative check."""
    try:
        apps_out = subprocess.run(
            ["nvidia-smi", f"--id={gpu_id}",
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        total = 0.0
        for line in apps_out.splitlines():
            parts = line.split(",")
            if len(parts) != 2:
                continue
            try:
                pid_val, mem_val = int(parts[0].strip()), float(parts[1].strip())
            except (ValueError, TypeError):
                continue
            try:
                with open(f"/proc/{pid_val}/cmdline", "rb") as f:
                    cmdline = f.read().decode("utf-8", errors="ignore")
                if "vllm" in cmdline.lower():
                    total += mem_val
            except (FileNotFoundError, PermissionError):
                pass
        return total
    except Exception:
        return 0.0


def _gpu_vllm_pids(gpu_id: int) -> "list[tuple[int, str]]":
    """[(pid, "<n> MiB")] for every vLLM process on the GPU (cmdline contains
    'vllm').  Unreadable / vanished processes and non-vLLM GPU users (embedding
    servers, notebooks) are skipped; nvidia-smi errors yield []."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=5,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    found: list[tuple[int, str]] = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if not parts or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        mem = parts[1] if len(parts) > 1 else "? MiB"
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
        except (FileNotFoundError, PermissionError):
            continue
        if "vllm" in cmdline.lower():
            found.append((pid, mem))
    return found


def _find_pid_on_port(port: int) -> int | None:
    """Return PID of the process listening on TCP port, or None."""
    try:
        result = subprocess.run(
            ["ss", "-tlnHp", "sport", f"= :{port}"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    import re
    m = re.search(r"pid=(\d+)", result.stdout)
    return int(m.group(1)) if m else None


def _find_vllm_pid_for_port(port: int) -> int | None:
    """Find a `vllm serve` or `vllm-omni serve` process targeting this port,
    whether the port is
    bound yet or not.  Lets adoption recognise vLLM instances that are still
    cold-starting (model load takes ~30-60s; the API port is not bound until
    then).  Prefers the listening PID if present, else falls back to pgrep
    over the cmdline."""
    if pid := _find_pid_on_port(port):
        return pid
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"vllm(-omni)? serve.* --port {port}( |$)"],
            capture_output=True, text=True, timeout=2,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    pids = [int(p) for p in result.stdout.split() if p.isdigit()]
    return pids[0] if pids else None


def _read_served_model_name(pid: int) -> str | None:
    """Extract --served-model-name from a process cmdline.  Lets adoption
    identify which configured model a running vLLM is serving without needing
    auth credentials to call /v1/models."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            args = f.read().split(b"\x00")
    except (FileNotFoundError, PermissionError):
        return None
    for i, a in enumerate(args):
        if a == b"--served-model-name" and i + 1 < len(args):
            return args[i + 1].decode("utf-8", errors="ignore")
    return None

@dataclass(frozen=True)
class ModelConfig:
    """Runtime and request contract for one dynamically scheduled model."""

    script: str
    served_name: str
    allowed_gpu_ids: set[int] | None = None
    request_kind: Literal["chat", "speech", "transcription"] = "chat"
    max_input_chars: int | None = None
    max_num_seqs: int | None = None
    # Scheduled by VRAM budget instead of taking a GPU slot: a small model that
    # shares a GPU with whichever primary model owns the slot.  It is admitted
    # only where the GPU's current free VRAM covers its min-viable footprint,
    # and it yields to primaries (see DynamicRouter.reclaim_coresident_for).
    coresident: bool = False
    # Seconds idle before unload; None keeps the global IDLE_TIMEOUT.
    idle_timeout: int | None = None

    def __post_init__(self) -> None:
        if self.idle_timeout is not None and (
            type(self.idle_timeout) is not int or self.idle_timeout <= 0
        ):
            raise ValueError("idle_timeout must be a positive integer")
        if self.request_kind != "chat":
            return
        if type(self.max_num_seqs) is not int or self.max_num_seqs <= 0:
            raise ValueError("chat model max_num_seqs must be a positive integer")


def _tts_max_input_chars() -> int:
    value = int(os.environ.get("QWEN3_TTS_MAX_INPUT_CHARS", "3000"))
    if value <= 0:
        raise ValueError("QWEN3_TTS_MAX_INPUT_CHARS must be greater than zero")
    return value


# Model configs. allowed_gpu_ids=None means the model may run on any GPU slot.
#
# 35B-A3B used to be pinned to GPU 1 because the embedding-provider on GPU 0
# took ~2.5–3.5 GiB, leaving only ~28.5 GiB free vs the 29.16 GiB needed for
# gpu_memory_utilization=0.93 × 32 GiB.  Relaxing to None now: _check_gpu_free
# guards against actually-too-tight cases at spawn time, and most of the time
# GPU 0 has enough headroom to host a scale-out 35B replica.
#
# The startup script receives VLLM_CUDA_DEVICE and VLLM_PORT from model_manager
# at spawn time.
MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Stock qwen3.6-35b-a3b retired 2026-09-01: the heretic abliteration serves
    # the same checkpoint lineage with identical tuning, and nothing needed the
    # censored variant.  Removed rather than aliased (same policy as the 27b
    # retirement) so the old name fails loudly.  Weights kept on disk.
    "qwen3.6-35b-a3b-heretic": ModelConfig(
        "run_qwen36_35b_heretic.sh", "qwen3.6-35b-a3b-heretic", max_num_seqs=16
    ),
    # Qwen3.8-27B: same qwen3_5 arch as the 3.6-27B (64 layers / 16 full-attn /
    # 4 KV heads / head_dim 256), so it shares the 3.6 slot's tuning verbatim.
    # Registered alongside rather than replacing it: both compete for the same
    # two slots via idle-unload, which keeps an A/B possible and the rollback
    # free.  Retire "qwen3.6-27b" once 3.8 has proven itself on real traffic.
    # qwen3.6-27b retired 2026-08-27: superseded by qwen3.8-27b, which is
    # lighter (16.74 vs 18.41 GiB) AND holds more KV (8.91 vs 7.25 GiB,
    # 200,118 vs 162,669 tok).  Removed rather than aliased so the old name
    # fails loudly and callers migrate explicitly.  Weights kept on disk.
    # max_num_seqs=8, validated by the single-variable A/B of 2026-09-03 (same
    # day, same traffic source, only the seq limit changed):
    #   seq=8 09:05-09:55: 491 reqs, lambda=10.1/min, E[S]=54s, offered load
    #     rho=9.0 concurrent; both instances ran 7+7 with waiting=0 throughout.
    #   seq=4 windows (09-02 all day + 09-03 10:05+): rho=9 > 2x4 capacity, so
    #     the queue is mathematically unstable at peak (observed waiting up to
    #     16) and the replica churns through cold starts.
    #   Decode throughput curve (1121 pure-decode samples): aggregate scales
    #     near-linearly 96->568 tok/s from 1->8 running; per-request speed is
    #     flat, 72.0 (batch 4) vs 71.0 (batch 8) tok/s, a 1.4% drop.  KV at 8:
    #     254,862/8 = 31.8k tokens per seq vs ~9k p50 requests.
    "qwen3.8-27b": ModelConfig(
        "run_qwen38_27b.sh", "qwen3.8-27b", max_num_seqs=8
    ),
    "qwen3-tts-1.7b-customvoice": ModelConfig(
        "run_qwen3_tts_1_7b_customvoice.sh",
        "qwen3-tts-1.7b-customvoice",
        request_kind="speech",
        max_input_chars=_tts_max_input_chars(),
    ),
    # Speech-to-text (POST /v1/audio/transcriptions), FP8 online-quantized.
    # Co-resident: ~3 GiB, so it shares a GPU with a primary instead of taking
    # a slot, and unloads after 5 idle minutes (uploads are rare and a cold
    # start is ~1 min, so a short residency costs little and frees the card).
    # 0.6B rather than 1.7B (2026-09-26): on a 34 min meeting it agrees with the
    # 1.7B BF16 at 0.954 (1.7B FP8: 0.971) and beats FunASR on product names
    # (Codex/connector/eval), while fitting beside a 27B where 1.7B (~7 GiB) does not.
    "qwen3-asr-0.6b": ModelConfig(
        "run_qwen3_asr_0_6b.sh",
        "qwen3-asr-0.6b",
        request_kind="transcription",
        coresident=True,
        idle_timeout=300,
    ),
}

# Minimum free GPU memory (GiB, from nvidia-smi) required to start a model.
# Used as a pre-eviction guard: before evicting an idle model to free a slot,
# check that the target GPU will have enough room after the eviction — this
# prevents destructively clearing a slot and then immediately failing the spawn.
# Rule of thumb: gpu_memory_utilization × GPU_total_GiB + 1 GiB safety buffer.
# If a model is not listed here no pre-check is performed (may evict & fail).
MODEL_MIN_FREE_GIB: dict[str, float] = {
    # Sized off the min-viable floor (0.84), not the preferred util (0.93):
    # this gate runs before the VRAM-aware util calculation, so keying it to the
    # preferred value rejected any card that could still host the model at the
    # floor.  0.84 × 31.8 GiB = 26.75 GiB, so 27.0 leaves 0.25 GiB of headroom —
    # deliberately thin, because every extra tenth here is a card this model can
    # no longer land on, and the util calculation downstream still refuses the
    # spawn if the floor genuinely does not fit.  A measured 27.29 GiB free (the
    # real contended-GPU-1 water line) has to pass: at 27.5 it did not.
    # Keep in step with MODEL_MIN_GPU_MEM_UTIL — if the floor moves and this does
    # not, the stricter of the two silently wins.
    "qwen3.6-35b-a3b-heretic": 27.0,
    # Measured 2026-08-27 on a free GPU 1 at util 0.84 / max-model-len 65536:
    #   weights 16.74 GiB + KV 6.65 GiB (149,796 tok, 2.29x concurrency)
    #   = 23.39 GiB of the 26.75 GiB budget; 3.36 GiB is activations+graphs.
    # MEASURED water line 2026-08-27 (filler tensor on GPU 1, direct vLLM spawn):
    #   free 24.21 GiB @ util 0.78 -> FAILED
    #   free 24.99 GiB @ util 0.78 -> FAILED
    #   free 25.48 GiB @ util 0.79 -> started
    #   free 26.26 GiB @ util 0.79 -> started
    # vLLM's own message gives the rule: it needs free >= util x 31.36 GiB (its
    # view of total) plus ~0.5 GiB already taken by its own CUDA context.  At the
    # 0.78 floor that is 24.96 GiB.  This gate must sit above what the util
    # calculation downstream will accept, which is 0.78 x 31.84 + 0.75 buffer =
    # 25.59 GiB — stricter than vLLM's hard limit, so 25.6 is the binding number
    # and anything higher only costs placements.  27.5 (copied from 3.6-27b) and
    # 27.0 (reasoned by analogy to the 35B) both rejected cards this model starts
    # on fine.  Re-measure if --max-model-len or MODEL_MIN_GPU_MEM_UTIL moves.
    "qwen3.8-27b":     25.6,
    # Measured on an RTX 5090: the two-stage vLLM-Omni process settles at
    # ~29.3 GiB after Code2Wav CUDA-graph capture.  Require another 1 GiB so a
    # partially occupied card is rejected before launch instead of failing OOM.
    "qwen3-tts-1.7b-customvoice": 30.3,
}

# Per-model preferred (maximum) gpu_memory_utilization — mirrors the run scripts.
# At spawn time model_manager reads the GPU's *current* free VRAM and lowers util
# to whatever actually fits, so a model can still start on a GPU that another
# process (e.g. the embedding-provider) is sharing.  util is only ever lowered,
# never raised above these tuned defaults.  vLLM treats --gpu-memory-utilization
# as a fraction of TOTAL memory and refuses to start when util×total exceeds the
# memory free at launch — that's the failure this avoids.
MODEL_GPU_MEM_UTIL: dict[str, float] = {
    "qwen3.6-35b-a3b-heretic": 0.93,
    # 0.84→0.90 (2026-09-03): ~29.7 GiB actual usage, ~+55k KV tokens (mostly
    # extra prefix-cache retention for the memory pipeline's repeated long
    # prompts).  Leaves ~2.9 GiB for neighbours — covers gliner (0.5) plus a
    # late-loading video-transcribe (1.6) with ~0.8 spare; 0.92 would not.
    # The spawn-time clamp below still lowers this to fit whatever is free.
    "qwen3.8-27b":             0.90,
    # PROVISIONAL (0.6B FP8, --enforce-eager, --max-model-len 2048): weights ~1.1
    # GiB + ~0.5 GiB CUDA context + activations/KV.  Not yet measured on a GPU
    # with room; the 1.7B measurement it replaces was 0.22 (weights 3.16 GiB).
    # Refine after the first real spawn (see the log's "Available KV cache").
    "qwen3-asr-0.6b":          0.10,
}
# Margin (MiB) held back from current free VRAM when computing util — absorbs
# nvidia-smi jitter and small growth by other GPU processes during vLLM startup.
GPU_MEM_UTIL_BUFFER_MIB = float(os.environ.get("GPU_MEM_UTIL_BUFFER_MIB", "768"))
# Slack insisted on between a model's min-viable reservation and free VRAM, so
# nvidia-smi jitter cannot turn a fitting spawn into a startup OOM.
HARD_MARGIN_MIB = 256.0
# Co-resident models get their own port lanes, far above the slot ports (a vLLM
# EngineCore binds api_port+2): lane port = base + gpu*100 + model_index*10.
CORESIDENT_PORT_BASE = int(os.environ.get("GPU_CORESIDENT_PORT_BASE", "9100"))

# Per-model MINIMUM viable gpu_memory_utilization.  vLLM allocates: weights +
# activations first, then *all remaining* budget (util×total − used) becomes the
# KV cache.  Lowering util shrinks the KV cache, not the weights.  The 35B models
# are hybrid Mamba+attention with a tiny KV cache even at 0.93 (~1.8 GiB); drop
# util a little and KV allocation hits zero → vLLM dies ~40 s into startup with
# "No available memory for the cache blocks".  So util can only be lowered within
# a narrow safe band.  Below the floor we DON'T spawn: we raise immediately so the
# failure is fast and legible (mirroring _check_gpu_free) instead of a slow OOM.
#
# Floors are empirical AND they move with --max-model-len.  The 0.90 here was
# measured when 35B ran at max-model-len 122880, where KV really did collapse
# just below it.  At the 81920 the run scripts now use it is far too
# conservative: 0.84 measured on GPU 1 — the contended card — gives 169182 KV
# tokens, 2.07x a full-length request and ~10x what max-num-seqs 16 needs at the
# ~6k-token median.  Verified serving a real completion, not merely starting.
#
# The floor doubles as the admission test, so setting it too high costs the whole
# card.  GPU 1 is shared with video-transcribe-service and an embedding provider
# (~3-4 GiB between them), leaving ~28 GiB free — just under 0.90's 28.9 GiB, so
# every spawn there was refused as "too tight" although the model fits fine.
# 0.84 needs 26.8 GiB and tolerates ~4.8 GiB of neighbours.
#
# Re-measure if --max-model-len changes again.  27B is dense with a normal KV
# cache and tolerates more headroom.
MODEL_MIN_GPU_MEM_UTIL: dict[str, float] = {
    "qwen3.6-35b-a3b-heretic": 0.84,
    "qwen3.8-27b":             0.78,
    # PROVISIONAL: a 2 GiB budget failed at startup on GPU0 (1.43 GiB usable
    # after the CUDA context), so the floor is above that.  Also the co-resident
    # admission floor (see _pick_coresident_gpu).
    "qwen3-asr-0.6b":          0.08,
}
# Fallback floor for models not listed above (best-effort attempt, not fail-fast).
GPU_MEM_UTIL_FLOOR = float(os.environ.get("GPU_MEM_UTIL_FLOOR", "0.78"))

# Per-model default --max-model-len — mirrors the run scripts.  Needed so the
# VRAM-aware max-len fallback knows the baseline before reducing it.  At the
# min-viable util a tight GPU may not hold the KV cache for this full context;
# vLLM then exits with "estimated maximum model length is N", and we retry the
# spawn once with VLLM_MAX_MODEL_LEN ≈ N so the model comes up at shorter context
# rather than failing.  (35B/heretic 32768, 27B 65536 — see the run scripts.)
MODEL_MAX_MODEL_LEN: dict[str, int] = {
    "qwen3.6-35b-a3b-heretic": 32768,
    "qwen3.8-27b":             65536,
}
# Don't bother reducing below this — a context this short is rarely useful, so we
# fail the spawn instead and let the model run elsewhere / retry when VRAM frees.
MIN_USEFUL_MODEL_LEN = int(os.environ.get("MIN_USEFUL_MODEL_LEN", "16384"))
# Safety haircut applied to vLLM's estimated-max-len before retrying, so we sit
# comfortably under the limit (KV must hold ≥1 full sequence plus a little slack).
MODEL_LEN_RETRY_FRACTION = float(os.environ.get("MODEL_LEN_RETRY_FRACTION", "0.92"))

_HOP_BY_HOP = frozenset({
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
# Suppress aiohttp internal connection-error tracebacks (ConnectionRefused during
# health polling is expected and caught by our code; no need to see it in logs).
logging.getLogger("aiohttp.client").setLevel(logging.CRITICAL)
logging.getLogger("aiohttp.connector").setLevel(logging.CRITICAL)


class GPUBusyError(Exception):
    """All GPU slots are occupied by other models."""
    pass


class BackendGoneError(Exception):
    """The vLLM backend disappeared BEFORE any response bytes were streamed to
    the client — connection refused/reset or ServerDisconnectedError during a
    crash-recycle (SIGKILL) or reload.  Raised (not returned) so `handle` can
    transparently respawn and retry: nothing has reached the client yet, so a
    retry is safe and turns a would-be 502 into either a 200 or a retryable
    503.  Mid-stream failures never raise this — they are caught where the
    response body is already being written."""
    pass


# ── Admin dashboard (served at GET /admin) ─────────────────────────────────────
ADMIN_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Gateway — slots</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.45 system-ui, sans-serif; margin: 0; padding: 24px;
         background:#0e1116; color:#e6edf3; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color:#8b949e; margin-bottom: 18px; }
  .grid { display:flex; flex-wrap:wrap; gap:16px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px;
          padding:16px; width:340px; }
  .row { display:flex; justify-content:space-between; margin:3px 0; }
  .k { color:#8b949e; }
  .badge { display:inline-block; padding:2px 9px; border-radius:999px;
           font-size:12px; font-weight:600; }
  .free    { background:#21262d; color:#8b949e; }
  .ready   { background:#1a7f37; color:#fff; }
  .starting{ background:#9e6a03; color:#fff; }
  .failed  { background:#b62324; color:#fff; }
  .model { font-size:15px; font-weight:600; margin:2px 0 10px; }
  select, button { font:inherit; padding:6px 10px; border-radius:6px;
                   border:1px solid #30363d; background:#21262d; color:#e6edf3; }
  button { cursor:pointer; }
  button.kill   { border-color:#b62324; color:#ff7b72; }
  button.start  { border-color:#1a7f37; color:#3fb950; }
  button.switch { border-color:#9e6a03; color:#d29922; }
  button:disabled { opacity:.4; cursor:not-allowed; }
  .ctl { display:flex; gap:8px; margin-top:10px; align-items:center; }
  .msg { color:#8b949e; font-size:12px; margin-top:10px; min-height:16px; }
  .meta { color:#6e7681; font-size:12px; margin-top:14px; }
</style>
</head>
<body>
<h1>LLM Gateway — GPU slots</h1>
<div class="sub">Auto-refresh every 2s · loopback admin only</div>
<div id="grid" class="grid"></div>
<div id="meta" class="meta"></div>
<script>
async function post(path, payload) {
  await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                     body: JSON.stringify(payload)});
  setTimeout(refresh, 300);
}
function opts(models, current) {
  return models.map(m => `<option value="${m}" ${m===current?'selected':''}>${m}</option>`).join('');
}
function card(s) {
  const occupied = s.model !== null;
  const sel = `<select id="m${s.slot_id}">${opts(s.allowed_models, s.model)}</select>`;
  let ctl;
  if (!occupied) {
    ctl = `${sel}<button class="start" onclick="start(${s.slot_id})">Start</button>`;
  } else {
    ctl = `${sel}`
        + `<button class="switch" onclick="switchM(${s.slot_id})">Switch</button>`
        + `<button class="kill" onclick="kill(${s.slot_id})">Kill</button>`;
  }
  const idle = s.idle_seconds==null ? '—' : s.idle_seconds + 's';
  return `<div class="card">
    <div class="row"><span class="k">slot ${s.slot_id} · GPU ${s.gpu_id} · :${s.port}</span>
      <span class="badge ${s.state}">${s.state}</span></div>
    <div class="model">${occupied ? s.model : '<i style="color:#6e7681">free</i>'}</div>
    <div class="row"><span class="k">active requests</span><span>${s.active_requests}</span></div>
    <div class="row"><span class="k">idle</span><span>${idle}</span></div>
    <div class="ctl">${ctl}</div>
    <div class="msg">${s.msg || ''}</div>
  </div>`;
}
function start(id)   { post('/admin/start',  {slot_id:id, model:document.getElementById('m'+id).value}); }
function switchM(id) { post('/admin/switch', {slot_id:id, model:document.getElementById('m'+id).value}); }
function kill(id)    { if (confirm('Kill vLLM on slot '+id+'?')) post('/admin/kill', {slot_id:id}); }
async function refresh() {
  try {
    const r = await fetch('/admin/status'); const d = await r.json();
    document.getElementById('grid').innerHTML = d.slots.map(card).join('');
    document.getElementById('meta').textContent =
      `models: ${d.models.join(', ')} · idle_timeout ${d.idle_timeout}s · wake_timeout ${d.wake_timeout}s`;
  } catch (e) { document.getElementById('meta').textContent = 'status error: ' + e; }
}
refresh(); setInterval(refresh, 2000);
</script>
</body>
</html>
"""

# Read-only status page — served at /status for the PUBLIC path route
# (llm.preseen.ai/status → :8002 via the preseen-gateway tunnel).  No controls
# and no /admin endpoints are reachable through that route: the page fetches
# /status.json (GET, state only).  Keep it that way — /admin has no auth and
# must stay LAN/Tailscale-only.
STATUS_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Gateway — status</title>
<style>
  :root { color-scheme: light dark; }
  body { font: 14px/1.45 system-ui, sans-serif; margin: 0; padding: 24px;
         background:#0e1116; color:#e6edf3; }
  h1 { font-size: 18px; margin: 0 0 4px; }
  .sub { color:#8b949e; margin-bottom: 18px; }
  .grid { display:flex; flex-wrap:wrap; gap:16px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:10px;
          padding:16px; width:340px; }
  .row { display:flex; justify-content:space-between; margin:3px 0; }
  .k { color:#8b949e; }
  .badge { display:inline-block; padding:2px 9px; border-radius:999px;
           font-size:12px; font-weight:600; }
  .free    { background:#21262d; color:#8b949e; }
  .ready   { background:#1a7f37; color:#fff; }
  .starting{ background:#9e6a03; color:#fff; }
  .failed  { background:#b62324; color:#fff; }
  .model { font-size:15px; font-weight:600; margin:2px 0 10px; }
  .meta { color:#6e7681; font-size:12px; margin-top:14px; }
</style>
</head>
<body>
<h1>LLM Gateway</h1>
<div class="sub">GPU slots — read-only</div>
<div id="grid" class="grid"></div>
<div id="meta" class="meta"></div>
<script>
function card(s) {
  const idle = s.idle_seconds==null ? '—' : s.idle_seconds + 's';
  return `<div class="card">
    <div class="row"><span class="k">slot ${s.slot_id} · GPU ${s.gpu_id}</span>
      <span class="badge ${s.state}">${s.state}</span></div>
    <div class="model">${s.model !== null ? s.model : '<i style="color:#6e7681">free</i>'}</div>
    <div class="row"><span class="k">active requests</span><span>${s.active_requests}</span></div>
    <div class="row"><span class="k">idle</span><span>${idle}</span></div>
  </div>`;
}
async function refresh() {
  try {
    const r = await fetch('/status.json'); const d = await r.json();
    document.getElementById('grid').innerHTML = d.slots.map(card).join('');
    document.getElementById('meta').textContent =
      `models: ${d.models.join(', ')} · idle_timeout ${d.idle_timeout}s`;
  } catch (e) { document.getElementById('meta').textContent = 'status error: ' + e; }
}
refresh(); setInterval(refresh, 3000);
</script>
</body>
</html>
"""


# ── GPU slot (physical resource) ───────────────────────────────────────────────

class GpuSlot:
    """Represents one physical GPU.  At most one GpuBackend lives here at a time."""

    def __init__(self, slot_id: int, gpu_id: int, port: int):
        self.slot_id  = slot_id
        self.gpu_id   = gpu_id
        self.port     = port
        # Set when a backend claims this slot (even before vLLM is started).
        # None means the slot is free.
        self.backend: "GpuBackend | None" = None
        # monotonic timestamp of the last self-heal recycle on this slot.
        # Lives on the slot (not the backend) so it survives the respawn and
        # rate-limits recycle→respawn thrash across backend generations.
        self.last_recycle: float = 0.0

    @property
    def is_free(self) -> bool:
        return self.backend is None

    @property
    def current_model(self) -> str | None:
        return self.backend.model_name if self.backend else None

    def __repr__(self) -> str:
        return (f"GpuSlot(id={self.slot_id} gpu={self.gpu_id} "
                f"port={self.port} model={self.current_model!r})")


# ── Per-(model, slot) subprocess controller ────────────────────────────────────

class GpuBackend:
    """Manages one vLLM subprocess: a specific model on a specific GPU slot."""

    def __init__(
        self,
        model_name: str,
        script: str,
        served_name: str,
        slot: GpuSlot,
        *,
        max_num_seqs: int | None = None,
        idle_timeout: int | None = None,
        coresident: bool = False,
    ):
        self.model_name  = model_name
        self.served_name = served_name
        self.slot        = slot
        self.vllm_port   = slot.port
        self.gpu_id      = slot.gpu_id
        self.vllm_base   = f"http://127.0.0.1:{slot.port}"
        self.script      = os.path.join(SCRIPT_DIR, script)
        self.max_num_seqs = max_num_seqs
        self.idle_timeout = idle_timeout
        self.coresident   = coresident
        safe             = model_name.replace(".", "_")
        self.log_path    = os.path.join(LOG_DIR, f"{safe}_slot{slot.slot_id}.log")
        self.log         = logging.getLogger(f"mgr.s{slot.slot_id}.{model_name}")

        # Use subprocess.Popen (not asyncio.create_subprocess_exec) so that
        # Python's exit doesn't auto-kill the child.  asyncio's subprocess
        # transport SIGKILLs the child when the event loop closes; Popen leaves
        # it alone, letting vLLM survive model_manager restarts (paired with
        # systemd KillMode=process and adopt_existing_backends on next start).
        self.process: subprocess.Popen | None = None
        # Set when this backend was adopted (not spawned by us) — we kill it
        # by PID since we don't have a Popen handle.
        self._adopted_pid: int | None = None
        self._ready           = False
        self._failed          = False   # permanently dead; don't retry on this object
        self.last_activity    = time.monotonic()
        self._active_requests = 0
        self._lock            = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        # Set by DynamicRouter after construction; used for replica-aware idle
        # timeout (count sibling instances of the same model).
        self.router: "DynamicRouter | None" = None
        # VRAM-aware max-model-len fallback: when a tight GPU can't hold the
        # KV cache for the full default context, vLLM exits at startup and tells
        # us the largest context that *does* fit.  We capture that here and retry
        # the spawn once with the reduced length so the model still comes up
        # (at shorter context) instead of failing outright.  None = use default.
        self._max_model_len_override: int | None = None
        # Consecutive upstream 5xx seen while proxying to this backend; reset on
        # any non-5xx response.  Drives self-heal recycle (see _forward).
        self._consecutive_5xx = 0

    def _base_idle_timeout(self) -> int:
        """Seconds idle before unload: the model's own override, else the global."""
        return self.idle_timeout or IDLE_TIMEOUT

    @property
    def is_running(self) -> bool:
        if self.process is not None:
            return self.process.poll() is None
        if self._adopted_pid is not None:
            try:
                os.kill(self._adopted_pid, 0)
                return True
            except ProcessLookupError:
                return False
        return False

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise aiohttp session and idle watchdog.  Does NOT spawn vLLM yet."""
        self._ensure_session()
        self._ensure_idle_task()
        self.log.info(
            f"Backend claimed slot {self.slot.slot_id} "
            f"(GPU={self.gpu_id} port={self.vllm_port})"
        )

    def _ensure_session(self) -> None:
        """(Re)create the aiohttp session if it was never opened or has been
        closed.  An idle-unload that races with a new request closes this
        backend's session before the request's spawn acquires the lock; without
        re-opening it every /health poll raises 'Session is closed' and the
        spawn falsely times out after WAKE_TIMEOUT, killing a healthy vLLM."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60),
                timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=None),
            )

    def _ensure_idle_task(self) -> None:
        """Restart the idle/crash watchdog if it has exited.  The idle-unload
        path returns from _idle_loop, so a revived backend would otherwise spawn
        vLLM with no supervision (no idle unload, no crash detection)."""
        if self._idle_task is None or self._idle_task.done():
            self._idle_task = asyncio.create_task(self._idle_loop())

    async def queue_depth(self) -> int:
        """vLLM's real internal queue (num_requests_waiting) for this backend.

        This is the count of requests vLLM cannot fit into its current batch
        (KV-cache full or max_num_seqs reached) — the true saturation signal.
        Returns 0 on any error so a flaky /metrics never blocks scale decisions.
        """
        if not self._ready or not self.is_running:
            return 0
        self._ensure_session()
        try:
            async with self._session.get(
                f"{self.vllm_base}/metrics",
                timeout=aiohttp.ClientTimeout(total=3),
            ) as r:
                if r.status != 200:
                    return 0
                text = await r.text()
        except Exception:
            return 0
        for line in text.splitlines():
            if line.startswith("vllm:num_requests_waiting"):
                try:
                    return int(float(line.rsplit(" ", 1)[1]))
                except (ValueError, IndexError):
                    return 0
        return 0

    async def stop(self) -> None:
        """Gracefully stop this backend and release its slot."""
        if self._idle_task:
            self._idle_task.cancel()
        async with self._lock:
            await self._kill_process_locked()
        await self._close_session()
        if self.slot.backend is self:
            self.slot.backend = None

    async def _close_session(self) -> None:
        """Close the aiohttp session if still open.  Safe to call multiple times."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Idle / dead-process watchdog ───────────────────────────────────────────

    async def _idle_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(30)

                # Backend was permanently failed before vLLM ever started
                if self._failed:
                    await self._close_session()
                    return

                # Detect unexpected vLLM crash (spawned or adopted)
                crashed = False
                if self.process is not None and self.process.poll() is not None:
                    crashed = True
                    self.log.warning(
                        f"vLLM exited unexpectedly (rc={self.process.returncode}) — freeing slot"
                    )
                elif self._adopted_pid is not None:
                    try:
                        os.kill(self._adopted_pid, 0)
                    except ProcessLookupError:
                        crashed = True
                        self.log.warning(
                            f"adopted vLLM pid={self._adopted_pid} gone — freeing slot"
                        )
                if crashed:
                    self._ready  = False
                    self._failed = True
                    # Kill orphan children before dropping the handle — EngineCore ignores
                    # SIGTERM and survives the APIServer exit, holding GPU memory.
                    if self.process is not None:
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    elif self._adopted_pid is not None:
                        try:
                            os.killpg(self._adopted_pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    self.process      = None
                    self._adopted_pid = None
                    if self.slot.backend is self:
                        self.slot.backend = None
                    await self._close_session()
                    return   # this backend object is dead

                if not self.is_running or self._active_requests > 0:
                    continue
                # Asymmetric scale-in: when this model runs on >1 slot, the
                # highest-slot instance is an "extra replica" and sheds early so
                # the borrowed slot returns to its evicted model promptly.  The
                # lowest-slot instance is the primary and keeps the full timeout.
                timeout = self._base_idle_timeout()
                is_replica = False
                if self.router is not None:
                    siblings = self.router._running_backends(self.model_name)
                    if len(siblings) > 1 and \
                       self is not min(siblings, key=lambda b: b.slot.slot_id):
                        timeout = REPLICA_IDLE_TIMEOUT
                        is_replica = True
                idle = time.monotonic() - self.last_activity
                if idle < timeout:
                    continue
                if is_replica:
                    # Don't reclaim a replica while the model has been busy
                    # recently.  Two separate guards, both needed:
                    #
                    #  1. Sustained quiet: the saturation sampler stamps every
                    #     sample that saw a queued request anywhere on this
                    #     model.  Reclaiming during a trough between bursts is
                    #     what caused the 23:26→23:29 churn on 2026-09-04, and
                    #     an instantaneous check cannot see it.
                    #  2. Instantaneous: a sibling queue that is non-empty right
                    #     now.  Those requests are already proxied into that
                    #     vLLM and cannot be rebalanced here.
                    last_busy = (
                        self.router._last_backlog_at.get(self.model_name)
                        if self.router is not None else None
                    )
                    if last_busy is not None:
                        quiet_for = time.monotonic() - last_busy
                        if quiet_for < REPLICA_QUIET_BEFORE_RECLAIM:
                            continue
                    try:
                        depths = await asyncio.gather(
                            *(b.queue_depth() for b in siblings if b is not self)
                        )
                        if sum(depths) > 0:
                            continue
                    except Exception:
                        pass   # can't read siblings — fall through to reclaim
                async with self._lock:
                    if not self.is_running or self._active_requests > 0:
                        continue
                    idle = time.monotonic() - self.last_activity
                    if idle >= timeout:
                        self.log.info(
                            f"Idle {int(idle)}s (timeout {timeout}s) — "
                            f"unloading {self.model_name}"
                        )
                        await self._kill_process_locked()
                        if self.slot.backend is self:
                            self.slot.backend = None
                        await self._close_session()
                        return   # slot is free; exit watchdog
        except asyncio.CancelledError:
            pass

    # ── Process lifecycle ──────────────────────────────────────────────────────

    def _check_gpu_free(self) -> None:
        """Raise RuntimeError if a LEFTOVER vLLM process is occupying this GPU's VRAM.

        A leftover is a vLLM process that no live backend of ours owns.  Live
        neighbours (a primary next to a co-resident model, or the reverse) are
        expected and skipped; other GPU users (e.g. embedding servers) are
        ignored because they share VRAM without the full allocation vLLM needs.

        vLLM's EngineCore and worker sub-processes can escape process-group kills
        and linger with large CUDA allocations.  Catching this before launching
        produces a clean error instead of an inscrutable OOM 60s into startup.
        """
        known = self.router.live_pgids() if self.router is not None else set()
        vllm_procs = [
            f"PID {pid} ({mem})"
            for pid, mem in _gpu_vllm_pids(self.gpu_id)
            if self._pgid_of(pid) not in known
        ]
        if not vllm_procs:
            return
        msg = (
            f"GPU {self.gpu_id} has leftover vLLM process(es) before spawn: "
            + ", ".join(vllm_procs)
            + ". Kill them manually or wait for them to exit, then retry."
        )
        self.log.error(msg)
        raise RuntimeError(msg)

    def _gpu_mem_util_for_spawn(self) -> float | None:
        """Compute a gpu_memory_utilization that fits this GPU's *current* free VRAM.

        Returns a value in [min_viable, default] to pass as VLLM_GPU_MEM_UTIL,
        falling back to the tuned default when VRAM can't be read.  Returns None
        only for models with no MODEL_GPU_MEM_UTIL entry (their run scripts do
        not read the variable).  The run scripts carry no default of their own —
        the manager is the single owner of this value.

        util is capped at the model's tuned default and only ever lowered to fit
        the GPU's current free VRAM, so a model can still start on a GPU shared
        with another process instead of OOM-ing on vLLM's "free memory < desired
        utilization" startup check.

        Crucially, util can only be lowered within a SAFE BAND.  vLLM gives all
        budget left after weights+activations to the KV cache, so lowering util
        shrinks the KV cache — and the hybrid Mamba+attention 35B models have a
        tiny KV cache even at their default, so a small drop starves it to zero
        and vLLM dies ~40 s in with "No available memory for the cache blocks".
        If the fit would fall below the model's min-viable util we raise here so
        the failure is fast and legible (mirroring _check_gpu_free) rather than a
        slow KV-allocation OOM after the weights have already loaded.
        """
        default = MODEL_GPU_MEM_UTIL.get(self.model_name)
        if default is None:
            return None
        total = _gpu_total_mib(self.gpu_id)
        free  = _gpu_free_mib(self.gpu_id)
        if not total or free is None:
            self.log.info(
                f"VRAM-aware util: could not read GPU {self.gpu_id} memory — "
                f"using tuned default util ({default})"
            )
            return default

        min_viable = MODEL_MIN_GPU_MEM_UTIL.get(self.model_name, GPU_MEM_UTIL_FLOOR)
        fit  = (free - GPU_MEM_UTIL_BUFFER_MIB) / total
        util = round(min(default, fit), 3)

        if util < min_viable:
            # The buffered fit dropped below the safe band.  Two cases:
            #
            #  (a) the floor itself STILL PHYSICALLY FITS in free VRAM — the only
            #      thing pushing us under was the comfort buffer.  Per "try hard to
            #      bring it up", clamp UP to the floor and spawn (best effort).  This
            #      is the common shared-GPU case: GPU 0 hosts the embedding-provider
            #      (~2-3 GiB), leaving ~29.3 GiB free → buffered fit 0.897 < 0.90,
            #      but 0.90×total still fits with room to spare.
            #
            #  (b) even the floor doesn't fit (would OOM ~40 s in at KV allocation).
            #      Raise like _check_gpu_free so the slot frees immediately and the
            #      caller retries/scales out once VRAM frees up.
            #
            # HARD_MARGIN is the minimum slack we insist on between the floor's raw
            # reservation and free VRAM, so a tiny nvidia-smi jitter doesn't OOM us.
            floor_needs_mib = min_viable * total
            if free >= floor_needs_mib + HARD_MARGIN_MIB:
                self.log.warning(
                    f"VRAM-aware util for {self.model_name}: GPU {self.gpu_id} "
                    f"free={free/1024:.1f} GiB / total={total/1024:.1f} GiB → buffered "
                    f"fit {util} below min-viable {min_viable}, but floor still fits "
                    f"({floor_needs_mib/1024:.1f} GiB + {HARD_MARGIN_MIB/1024:.1f} GiB "
                    f"margin ≤ free) — clamping up to {min_viable} (tight, best effort)"
                )
                return min_viable
            needed_mib = floor_needs_mib + HARD_MARGIN_MIB
            msg = (
                f"GPU {self.gpu_id} only {free/1024:.1f} GiB free — too tight for "
                f"{self.model_name}: even min-viable util {min_viable} "
                f"({floor_needs_mib/1024:.1f} GiB) won't fit (KV cache would be "
                f"starved). Need ≥{needed_mib/1024:.1f} GiB free. Not spawning; "
                f"will retry when VRAM frees up."
            )
            self.log.error(msg)
            raise RuntimeError(msg)

        if util < default:
            self.log.info(
                f"VRAM-aware util for {self.model_name}: GPU {self.gpu_id} "
                f"free={free/1024:.1f} GiB / total={total/1024:.1f} GiB "
                f"→ util={util} (default {default}, min-viable {min_viable}, "
                f"buffer {GPU_MEM_UTIL_BUFFER_MIB/1024:.1f} GiB)"
            )
        return util

    def _scan_kv_max_model_len(self) -> int | None:
        """If the last spawn died because the KV cache couldn't hold the full
        context, return vLLM's estimate of the largest context that *would* fit.

        vLLM's _check_enough_kv_cache_memory raises a ValueError whose message
        ends with "the estimated maximum model length is N." — we parse N from
        the tail of the spawn log.  Returns None if that pattern isn't present
        (i.e. the spawn failed for some other reason — don't second-guess it).
        """
        try:
            with open(self.log_path, "rb") as f:
                # Only the tail matters; the error is near the end of startup.
                try:
                    f.seek(-65536, os.SEEK_END)
                except OSError:
                    f.seek(0)
                tail = f.read().decode("utf-8", errors="ignore")
        except OSError:
            return None
        m = re.findall(r"estimated maximum model length is (\d+)", tail)
        if not m:
            return None
        return int(m[-1])

    async def _spawn_locked(self) -> None:
        """Spawn vLLM subprocess and wait for /health.  Caller must hold self._lock.

        On a tight GPU the model may load but fail because the KV cache can't hold
        the full default context.  We retry the spawn once with a reduced
        --max-model-len (VLLM_MAX_MODEL_LEN) taken from vLLM's own estimate, so the
        model comes up at shorter context instead of failing outright.
        """
        self._ready = False
        # ── Revival guard ──────────────────────────────────────────────────────
        # This backend object can be revived after an idle-unload that raced with
        # a new request: while this coroutine was blocked on the lock, the idle
        # watchdog killed the old vLLM, detached the slot, closed our aiohttp
        # session, and exited.  Re-establish all three before spawning — otherwise
        # every /health poll raises "Session is closed", the spawn times out after
        # WAKE_TIMEOUT, and we kill a perfectly healthy vLLM (→ 503 to the caller).
        self._ensure_session()
        self._ensure_idle_task()
        if self.slot.backend is None:
            self.slot.backend = self
        # Primaries outrank co-resident models for VRAM: stop an idle one on this
        # GPU if the primary would otherwise be too tight.
        if not self.coresident and self.router is not None:
            await self.router.reclaim_coresident_for(self)
        self._check_gpu_free()
        # Re-evaluate context length from scratch each wake: GPU free VRAM
        # fluctuates (the embedding-provider idle-offloads), so a backend revived
        # when its GPU has room should try the full default context again rather
        # than stay capped at a reduction computed during an earlier tight spawn.
        self._max_model_len_override = None
        # VRAM-aware gpu_memory_utilization computed once up front (raises if even
        # the min-viable util doesn't fit, so the slot frees immediately).
        util = self._gpu_mem_util_for_spawn()

        # At most 2 attempts: the initial spawn, plus one retry with a reduced
        # max-model-len if the KV cache couldn't hold the full context.
        for attempt in range(2):
            if await self._spawn_attempt_locked(util):
                return
            # _spawn_attempt_locked returned False ⇒ process exited.  See whether
            # it was the KV-can't-hold-full-context case and we can shrink to fit.
            if attempt == 0 and self._max_model_len_override is None:
                est = self._scan_kv_max_model_len()
                if est is not None:
                    default_len = MODEL_MAX_MODEL_LEN.get(self.model_name)
                    reduced = int(est * MODEL_LEN_RETRY_FRACTION)
                    # Round down to a multiple of 256 for tidy block alignment.
                    reduced -= reduced % 256
                    if reduced >= MIN_USEFUL_MODEL_LEN and (
                        default_len is None or reduced < default_len
                    ):
                        self._max_model_len_override = reduced
                        self.log.warning(
                            f"KV cache too small for full context on GPU "
                            f"{self.gpu_id}; vLLM estimates max len {est}. Retrying "
                            f"with --max-model-len {reduced} "
                            f"(default {default_len}) — reduced context, best effort."
                        )
                        # Let the SIGKILL'd process group fully release its CUDA
                        # memory before respawning, so the retry sees the freed VRAM.
                        await asyncio.sleep(3)
                        continue
                    self.log.error(
                        f"KV cache too small and reduced len {reduced} < "
                        f"{MIN_USEFUL_MODEL_LEN} (or ≥ default) — not retrying."
                    )
            raise RuntimeError(
                f"vLLM for '{self.model_name}' exited during startup on slot "
                f"{self.slot.slot_id}. See {self.log_path}."
            )

    async def _spawn_attempt_locked(self, util: float | None) -> bool:
        """One spawn + health-wait cycle.  Returns True if vLLM became healthy,
        False if the process exited (caller decides whether to retry).  Raises on
        watchdog-clear or startup timeout.  Caller must hold self._lock."""
        self._ready = False
        log_fd = open(self.log_path, "ab")
        try:
            self.log.info(
                f"Spawning vLLM for {self.model_name} "
                f"(GPU={self.gpu_id} port={self.vllm_port})"
            )
            spawn_env = {
                **os.environ,
                "VLLM_CUDA_DEVICE": str(self.gpu_id),
                "VLLM_PORT":        str(self.vllm_port),
                # Cap ninja parallelism for FlashInfer's JIT.  Unset it runs
                # `ninja -j $(nproc)` = 32 nvcc/cicc peaking near 50 GiB RSS,
                # which starves sshd/tailscaled/cloudflared and takes the box to
                # load 500+.  gateway.env normally sets this, but that file is
                # gitignored — defaulting here keeps the guard in tracked code so
                # a fresh deploy can't silently lose it.
                "MAX_JOBS": os.environ.get("MAX_JOBS", "4"),
            }
            if self.max_num_seqs is not None:
                spawn_env["VLLM_MAX_NUM_SEQS"] = str(self.max_num_seqs)
            # VRAM-aware gpu_memory_utilization: lower it to fit current free VRAM
            # so a shared GPU can still host the model instead of OOM-ing.
            if util is not None:
                spawn_env["VLLM_GPU_MEM_UTIL"] = f"{util:.3f}"
            # VRAM-aware max-model-len: set on a retry when the full context's KV
            # cache didn't fit (the run scripts read VLLM_MAX_MODEL_LEN).
            if self._max_model_len_override is not None:
                spawn_env["VLLM_MAX_MODEL_LEN"] = str(self._max_model_len_override)
            # nice/ionice so a spawn can never starve sshd/tailscaled/cloudflared.
            # The children inherit both, which is the point: on a cold JIT cache
            # FlashInfer compiles CUTLASS through a swarm of nvcc/cicc (see
            # MAX_JOBS in gateway.env), and unthrottled that swarm took the box to
            # load 500+ and dropped the Cloudflare tunnel.  Costs vLLM a little
            # CPU priority once warm, which is the trade we want under contention.
            self.process = subprocess.Popen(
                ["nice", "-n", "10", "ionice", "-c2", "-n7", "bash", self.script],
                stdout=log_fd, stderr=log_fd,
                env=spawn_env,
                start_new_session=True,
            )
        finally:
            log_fd.close()

        # Save the pid immediately after Popen so we can kill the process group
        # even if _idle_loop races and clears self.process before we detect the
        # failure.  EngineCore (a subprocess) inherits the same pgid and ignores
        # SIGTERM, so we must use SIGKILL to reclaim GPU memory on any failure.
        spawn_pid = self.process.pid

        deadline = time.monotonic() + WAKE_TIMEOUT
        started  = time.monotonic()
        while time.monotonic() < deadline:
            # Guard against _idle_loop clearing self.process concurrently
            # (it runs without the lock; we're inside the lock but yield at await).
            if self.process is None or self._failed:
                # Orphan-kill: _idle_loop dropped the handle but EngineCore
                # may still be running with 29+ GiB of GPU memory.
                try:
                    os.killpg(spawn_pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                raise RuntimeError(
                    f"vLLM for '{self.model_name}' crashed during startup "
                    f"(watchdog cleared process). See {self.log_path}."
                )
            rc = self.process.poll()
            if rc is not None:
                # APIServer exited — SIGKILL the process group immediately.
                # vLLM EngineCore subprocesses ignore SIGTERM and would otherwise
                # hold GPU memory until the next spawn attempt triggers a "leftover
                # process" error.
                try:
                    os.killpg(spawn_pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                self.process = None
                self.log.warning(
                    f"vLLM for '{self.model_name}' exited (rc={rc}) on slot "
                    f"{self.slot.slot_id}. See {self.log_path}."
                )
                return False
            try:
                async with self._session.get(
                    f"{self.vllm_base}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as r:
                    if r.status == 200:
                        elapsed = int(time.monotonic() - started)
                        self.log.info(
                            f"vLLM ready in {elapsed}s "
                            f"(slot {self.slot.slot_id} GPU={self.gpu_id})"
                        )
                        self._ready = True
                        # Idle counts from readiness, not from claim/spawn —
                        # otherwise a ~60s cold start eats half a replica's
                        # 120s idle allowance before it can serve anything.
                        self.last_activity = time.monotonic()
                        return True
                    else:
                        self.log.warning(
                            f"health poll: HTTP {r.status} after "
                            f"{int(time.monotonic()-started)}s"
                        )
            except Exception as exc:
                self.log.warning(
                    f"health poll: {type(exc).__name__}: {exc} after "
                    f"{int(time.monotonic()-started)}s"
                )
            await asyncio.sleep(HEALTH_POLL)

        self.log.error(f"Startup timed out after {WAKE_TIMEOUT}s — killing")
        await self._kill_process_locked()
        raise RuntimeError(
            f"vLLM for '{self.model_name}' did not become healthy within {WAKE_TIMEOUT}s. "
            f"See {self.log_path}."
        )

    async def _kill_process_locked(self) -> None:
        """Kill vLLM and its children (including orphan sub-processes like EngineCore).
        Handles both subprocess-owned and adopted backends — both have pgid==pid
        thanks to start_new_session=True.  Caller must hold self._lock."""
        self._ready = False
        pid = self.process.pid if self.process is not None else self._adopted_pid
        if pid is None:
            return
        # Always attempt to kill the entire process group, even if the APIServer
        # has already exited — orphan children (e.g. vLLM EngineCore) may still
        # hold ports or GPU memory and need to be explicitly reaped.
        self.log.info(f"Sending SIGTERM to pgid {pid}")
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            self.process = None
            self._adopted_pid = None
            return
        # Poll for death up to 30s, then escalate to SIGKILL
        for _ in range(30):
            await asyncio.sleep(1)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
        else:
            self.log.warning("SIGTERM timeout — sending SIGKILL")
            try:
                os.killpg(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        # EngineCore ignores SIGTERM and may outlive the APIServer (which dies
        # quickly, causing the poll above to exit early via ProcessLookupError).
        # Always sweep the GPU for orphan processes so the next spawn isn't
        # blocked by a zombie holding VRAM.  Scoped to our own process group so
        # we never touch another service sharing this GPU.
        await self._kill_gpu_zombies(pid)
        self.log.info("vLLM unloaded")
        self.process = None
        self._adopted_pid = None

    async def _ensure_running(self) -> None:
        """Block until this backend's vLLM is ready.  Serialised per-backend."""
        if self._failed:
            raise RuntimeError(
                f"Backend for '{self.model_name}' on slot {self.slot.slot_id} "
                f"has permanently failed — see {self.log_path}."
            )
        if self._ready and self.is_running:
            return
        async with self._lock:
            if self._failed:
                raise RuntimeError(
                    f"Backend for '{self.model_name}' on slot {self.slot.slot_id} "
                    f"has permanently failed — see {self.log_path}."
                )
            if self._ready and self.is_running:
                return
            try:
                await self._spawn_locked()
            except Exception:
                self._failed = True
                raise

    # ── Request proxying ───────────────────────────────────────────────────────

    async def proxy(self, request: web.Request, body: bytes) -> web.StreamResponse:
        self._active_requests += 1
        self.last_activity = time.monotonic()
        try:
            return await self._forward(request, body)
        finally:
            self._active_requests -= 1
            self.last_activity = time.monotonic()

    @staticmethod
    def _pgid_of(pid: int) -> int | None:
        """Process group of pid, or None if it is gone / unreadable.

        Field 5 of /proc/<pid>/stat.  Field 2 (comm) is parenthesised and may
        itself contain spaces, so split after the closing paren."""
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                data = fh.read()
            return int(data[data.rindex(b")") + 2:].split()[2])
        except (OSError, ValueError, IndexError):
            return None

    async def _kill_gpu_zombies(self, owner_pgid: int | None) -> None:
        """Reap leftover CUDA processes belonging to OUR vLLM process group.

        vLLM's EngineCore is a subprocess that ignores SIGTERM and can outlive
        the APIServer while still holding ~29 GiB of VRAM, so a post-crash sweep
        is genuinely needed.  But it has to be scoped to processes we started:
        these GPUs are shared with other services (the embedding provider on
        :7997, transcription jobs, ad-hoc notebooks), and an unscoped sweep
        SIGKILLs them.  It previously did exactly that — logging
        "Killing GPU 1 zombie PID 77589 (post-crash)" while killing an unrelated
        video-transcribe-service process.

        Ownership test is the process group: _spawn uses start_new_session=True,
        so our APIServer is a session leader with pgid == pid and every child
        (EngineCore included) inherits that pgid.  Anything on this GPU with a
        different pgid is somebody else's and is left alone."""
        if owner_pgid is None:
            return
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid",
                    "--format=csv,noheader",
                    f"--id={self.gpu_id}",
                ],
                capture_output=True, text=True, timeout=5,
            )
            pids = [
                int(l.strip())
                for l in result.stdout.splitlines()
                if l.strip().isdigit()
            ]
            for pid in pids:
                pgid = self._pgid_of(pid)
                if pgid != owner_pgid:
                    self.log.debug(
                        f"GPU {self.gpu_id} PID {pid} (pgid={pgid}) is not ours "
                        f"(pgid={owner_pgid}) — leaving it alone"
                    )
                    continue
                self.log.warning(
                    f"Killing GPU {self.gpu_id} orphan PID {pid} "
                    f"from our pgid {owner_pgid} (post-crash)"
                )
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
        except Exception as exc:
            self.log.warning(f"GPU zombie cleanup failed: {exc}")

    def _trip_dead(self, reason: str) -> None:
        """Mark this backend dead, detach its slot, and SIGKILL the process group
        so the GPU frees and the next request spawns a fresh instance.  Used by
        the self-heal path (repeated upstream 5xx) and EngineCore crash detection.
        Records the recycle time on the SLOT so RECYCLE_COOLDOWN survives respawn."""
        self.log.error(
            f"Recycling backend on slot {self.slot.slot_id} (GPU={self.gpu_id}): {reason}"
        )
        self._ready  = False
        self._failed = True
        self.slot.last_recycle = time.monotonic()
        if self.slot.backend is self:
            self.slot.backend = None
        # SIGKILL the whole group — EngineCore ignores SIGTERM and would otherwise
        # linger holding ~29 GiB of GPU memory.
        pid = self.process.pid if self.process is not None else self._adopted_pid
        if pid is not None:
            try:
                os.killpg(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        self.process = None
        self._adopted_pid = None
        # Belt-and-suspenders: reap any orphan vLLM procs still on this GPU.
        # pid is captured above because self.process is already cleared by the
        # time this task runs, and without it the sweep has no ownership test.
        asyncio.create_task(self._kill_gpu_zombies(pid))

    async def _forward(self, request: web.Request, body: bytes) -> web.StreamResponse:
        target_url  = f"{self.vllm_base}{request.path_qs}"
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP | {"host", "content-length"}
        }
        # Normalise the body before it reaches vLLM.  Parsed once, re-serialised
        # only if something actually changed.
        if body:
            try:
                parsed = json.loads(body)
            except (json.JSONDecodeError, TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict):
                changed = False
                # Rewrite routing key → vLLM's own --served-model-name
                if (self.served_name != self.model_name
                        and parsed.get("model") == self.model_name):
                    parsed["model"] = self.served_name
                    changed = True
                # LiteLLM's Responses API → chat/completions conversion always
                # emits a `tools` key, so a caller that passed no tools at all
                # still arrives here as `tools: []`.  vLLM rejects that outright
                # ("`tools` must not be an empty array. Either provide at least
                # one tool or omit the field entirely"), which 400s every
                # tool-less /v1/responses request — that is what breaks Memory
                # extraction.  Strip it here rather than in LiteLLM: this is the
                # last hop before vLLM, so Open WebUI and direct callers are
                # covered by the same fix.
                if parsed.get("tools") == []:
                    parsed.pop("tools", None)
                    # tool_choice without tools is meaningless, and vLLM rejects
                    # "required" with no tools to choose from.
                    parsed.pop("tool_choice", None)
                    changed = True
                if changed:
                    body = json.dumps(parsed).encode()
        try:
            async with self._session.request(
                method=request.method, url=target_url,
                headers=fwd_headers, data=body,
            ) as upstream:
                # ── Self-heal: recycle a ready-but-degraded backend on 5xx ───
                # A 5xx from a *ready* vLLM means the instance is broken, not the
                # request: EngineCore crash ("EngineCore encountered an issue"),
                # CUDA error, or a wedged scheduler.  EngineCore subprocesses often
                # survive the API-server exit and hold ~29 GiB of GPU memory, so we
                # must kill the group, not just drop the handle.
                #   • EngineCore crash → recycle on the first hit (known fatal).
                #   • Other 5xx        → recycle after RECYCLE_5XX_THRESHOLD in a
                #     row, gated by a slot-level cooldown so a request-driven 500
                #     loop can't thrash the GPU (each respawn ≈ 90s cold start).
                if upstream.status >= 500:
                    err_body = await upstream.read()
                    is_enginecore = b"EngineCore" in err_body
                    self._consecutive_5xx += 1
                    cooling = (time.monotonic() - self.slot.last_recycle) < RECYCLE_COOLDOWN
                    if is_enginecore:
                        self._trip_dead("vLLM EngineCore crash during inference")
                    elif self._consecutive_5xx >= RECYCLE_5XX_THRESHOLD and not cooling:
                        self._trip_dead(
                            f"{self._consecutive_5xx} consecutive upstream "
                            f"{upstream.status} responses"
                        )
                    elif self._consecutive_5xx >= RECYCLE_5XX_THRESHOLD:
                        self.log.warning(
                            f"slot {self.slot.slot_id}: {self._consecutive_5xx} "
                            f"consecutive {upstream.status} but within "
                            f"{RECYCLE_COOLDOWN}s recycle cooldown — not respawning"
                        )
                    return web.Response(
                        status=upstream.status, content_type="application/json",
                        body=err_body,
                    )
                # Healthy response — clear the consecutive-5xx streak.
                self._consecutive_5xx = 0
                # ── Normal streaming path ───────────────────────────────────
                resp_headers = {
                    k: v for k, v in upstream.headers.items()
                    if k.lower() not in _HOP_BY_HOP | {"content-length"}
                }
                resp = web.StreamResponse(status=upstream.status, headers=resp_headers)
                try:
                    await resp.prepare(request)
                    async for chunk in upstream.content.iter_any():
                        await resp.write(chunk)
                    await resp.write_eof()
                except Exception as exc:
                    self.log.debug(f"Stream interrupted: {exc}")
                return resp
        except aiohttp.ClientConnectorError as exc:
            # Backend port not accepting connections — killed mid-flight or not
            # yet bound.  We are still pre-stream here (the streaming body has
            # its own except above), so let handle() respawn and retry.
            self.log.warning(f"Cannot reach vLLM (pre-stream): {exc}")
            raise BackendGoneError(f"connect: {exc}") from exc
        except aiohttp.ClientError as exc:
            # ServerDisconnectedError et al: the backend dropped the connection
            # before sending a response (typically a crash-recycle SIGKILL).
            # Previously returned a hard 502; instead signal a retryable gone.
            self.log.warning(f"Backend disconnected (pre-stream): {exc}")
            raise BackendGoneError(str(exc)) from exc


# ── Task-id affinity (for clients that propagate `x-task-id` header) ───────────
_AFFINITY_MAX = 10000
_task_affinity: "OrderedDict[str, int]" = OrderedDict()


def _set_task_affinity(task_id: str, slot_id: int) -> None:
    _task_affinity[task_id] = slot_id
    _task_affinity.move_to_end(task_id)
    while len(_task_affinity) > _AFFINITY_MAX:
        _task_affinity.popitem(last=False)


def _flatten_user_text(content) -> str | None:
    """messages[i].content (chat/completions) or input[i].content (responses)
    can be a plain string OR a list of dicts (multimodal parts).  Return the
    first text chunk we find, or None."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text") or part.get("content")
                if isinstance(txt, str):
                    return txt
    return None


_STICKY_HASH_MIN_BLOB = 50  # below this, the content is too short to bother
                            # (e.g. "ping" probes); skip sticky and let
                            # least-connections handle them.


def _content_sticky_key(parsed_body: dict) -> str | None:
    """Pipeline-agnostic sticky key: hash of system + first user message.

    Same logical task across multi-turn re-attempts: messages[1] (the original
    user turn) doesn't change when later turns (assistant + QC feedback) get
    appended, so the hash is stable.

    Distinct tasks: the first user message differs → distinct hashes →
    natural load balancing across slots.

    Works uniformly for /v1/chat/completions (messages[]) and /v1/responses
    (instructions + input, where `input` may be a string or list).
    """
    parts: list[tuple[str, str]] = []

    # chat/completions: walk messages, collect at most system + first user
    for m in (parsed_body.get("messages") or []):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role == "system":
            if (txt := _flatten_user_text(m.get("content"))):
                parts.append(("system", txt))
        elif role == "user":
            if (txt := _flatten_user_text(m.get("content"))):
                parts.append(("user", txt))
            break  # stop at first user — later turns are per-attempt noise

    # responses API: instructions + first user from input
    if not parts:
        ins = parsed_body.get("instructions")
        if isinstance(ins, str) and ins:
            parts.append(("system", ins))
        inp = parsed_body.get("input")
        if isinstance(inp, str) and inp:
            parts.append(("user", inp))
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, dict) and item.get("role") == "user":
                    if (txt := _flatten_user_text(item.get("content"))):
                        parts.append(("user", txt))
                    break

    if not parts:
        return None
    blob = "\n".join(f"{r}:{c}" for r, c in parts)
    if len(blob) < _STICKY_HASH_MIN_BLOB:
        return None
    # 16-hex-char key from blake2b-64.  Prefix "h:" so logs distinguish
    # content-derived sticky keys from real task_ids when grepping.
    return "h:" + hashlib.blake2b(
        blob.encode("utf-8", errors="replace"), digest_size=8
    ).hexdigest()


def _extract_task_id(parsed_body: dict | None, headers) -> str | None:
    """Locate a stable per-task identifier across several places, in priority order:
      1. `x-task-id` HTTP header — cleanest, but LiteLLM strips client headers
         when proxying via openai-python SDK, so this rarely arrives.
      2. `metadata.user_id.device_id` in body — annotation-pipeline path.
         Field is named "device_id" but value is the per-task identifier
         injected by claude CLI via .claude.json:userID.
      3. `metadata.task_id` in body — generic LiteLLM metadata passthrough.
      4. `user` field in body — OpenAI-standard; legacy path.
      5. Content-derived hash: blake2b of system + first user message.
         Pipeline-agnostic fallback; works for any request shape whose task
         identity sits in the first user turn (the typical agent layout).
         Keys returned by this path are prefixed "h:" so they're greppable.
    """
    if tid := (headers.get("x-task-id") or headers.get("X-Task-Id")):
        return tid
    if parsed_body:
        meta = parsed_body.get("metadata")
        if isinstance(meta, dict):
            user_id = meta.get("user_id")
            if isinstance(user_id, dict) and (tid := user_id.get("device_id")):
                return str(tid)
            if tid := meta.get("task_id"):
                return str(tid)
        if tid := parsed_body.get("user"):
            return str(tid)
        if tid := _content_sticky_key(parsed_body):
            return tid
    return None


def _sticky_slot_for(parsed_body: dict | None, headers) -> tuple[int | None, str]:
    """Resolve the preferred slot for this request via x-task-id.
    Returns (slot_id | None, reason)."""
    task_id = _extract_task_id(parsed_body, headers)
    if task_id and (s := _task_affinity.get(task_id)) is not None:
        return s, f"task:{task_id}"
    return None, "none"


# ── Dynamic Router ─────────────────────────────────────────────────────────────

class DynamicRouter:
    """Routes requests to GPU backends with dynamic slot assignment and scale-out."""

    # Scale-out is driven by the background saturation monitor
    # (_saturation_loop), which fires on a weighted backlog accumulator over
    # vLLM's real queue (num_requests_waiting) per SCALE_OUT_TIERS — see the
    # module constants.  There is no instantaneous in-flight-count trigger.

    # Seconds to wait before re-attempting scale-out after a failure.
    # Prevents a crash-loop when a slot cannot physically start a model
    # (e.g. not enough free VRAM because another process is on that GPU).
    SCALE_OUT_COOLDOWN = 600  # 10 minutes — back off longer after a failed scale-out

    def __init__(self, slots: list[GpuSlot], model_configs: dict[str, ModelConfig]):
        self.slots         = slots
        self.model_configs = model_configs
        self.log           = logging.getLogger("mgr.router")
        self._router_lock  = asyncio.Lock()   # serialises slot claims
        self._scale_tasks: set[asyncio.Task] = set()
        # Per-model timestamp of last scale-out failure (monotonic clock).
        # Used to enforce SCALE_OUT_COOLDOWN before retrying.
        self._scale_fail_time: dict[str, float] = {}
        # Per-slot last admin-action message, surfaced on the /admin dashboard
        # (e.g. "starting qwen3.6-27b…", "kill failed: …").
        self._admin_msg: dict[int, str] = {}
        # Background admin tasks (start/kill/switch), kept referenced so the
        # event loop doesn't GC them mid-flight.
        self._admin_tasks: set[asyncio.Task] = set()
        # Per-model scale-out accrual samples: deque of (monotonic_ts, increment).
        # P is the sum of increments newer than SCALE_WINDOW; fires at P≥1 (see
        # _saturation_loop / SCALE_OUT_TIERS).  In-memory only — a restart resets it.
        self._scale_accrual: dict[str, deque[tuple[float, float]]] = {}
        # Monotonic time of the last sample where a model had ANY queued
        # request across all its instances.  Scale-in reads this so a replica
        # is not reclaimed during a lull between bursts (see _idle_loop).
        self._last_backlog_at: dict[str, float] = {}
        # Monotonic time of the previous saturation sample, for the accrual dt.
        self._sat_last_tick: float = 0.0
        self._sat_task: asyncio.Task | None = None
        # Dedup for scale-out decline logging: (reason, monotonic) per model, so a
        # persistent blocker (e.g. GPU busy) logs once per window, not every cycle.
        self._scale_decline_log: dict[str, tuple[str, float]] = {}
        # Co-resident models live in "lanes": one virtual GpuSlot per
        # (model, gpu), created on demand.  Lanes are NOT in self.slots, so a
        # co-resident backend never occupies a primary slot and primaries never
        # see it as a claim; placement is decided by VRAM budget instead.
        self._lanes: dict[tuple[str, int], GpuSlot] = {}

    # ── Slot / backend helpers ─────────────────────────────────────────────────

    def _make_backend(self, model_name: str, slot: GpuSlot, *,
                      served_name: str | None = None) -> GpuBackend:
        """Build a backend for model_name on slot, wired to this router."""
        config = self.model_configs[model_name]
        b = GpuBackend(
            model_name,
            config.script,
            served_name or config.served_name,
            slot,
            max_num_seqs=config.max_num_seqs,
            idle_timeout=config.idle_timeout,
            coresident=config.coresident,
        )
        b.router = self
        return b

    def _all_slots(self) -> list[GpuSlot]:
        """Primary slots plus co-resident lanes."""
        return [*self.slots, *self._lanes.values()]

    def _coresident_names(self) -> list[str]:
        return sorted(n for n, c in self.model_configs.items() if c.coresident)

    def _lane_port(self, model_name: str, gpu_id: int) -> int:
        idx = self._coresident_names().index(model_name)
        return CORESIDENT_PORT_BASE + gpu_id * 100 + idx * 10

    def _lane_slot(self, model_name: str, gpu_id: int) -> GpuSlot:
        """The lane for (model, gpu): stable id and port, so a manager restart can
        find and adopt the vLLM it left running."""
        lane = self._lanes.get((model_name, gpu_id))
        if lane is None:
            idx = self._coresident_names().index(model_name)
            lane = GpuSlot(
                100 + gpu_id * 10 + idx, gpu_id, self._lane_port(model_name, gpu_id),
            )
            self._lanes[(model_name, gpu_id)] = lane
        return lane

    def _coresident_backends_on(self, gpu_id: int) -> list[GpuBackend]:
        return [
            lane.backend for (_m, g), lane in self._lanes.items()
            if g == gpu_id and lane.backend is not None and not lane.backend._failed
        ]

    def live_pgids(self) -> set[int]:
        """Process groups of every running backend (primary or co-resident).
        _check_gpu_free treats these as expected neighbours, not leftovers."""
        pgids: set[int] = set()
        for slot in self._all_slots():
            b = slot.backend
            if b is None or not b.is_running:
                continue
            pid = b.process.pid if b.process is not None else b._adopted_pid
            if pid is not None:
                pgids.add(GpuBackend._pgid_of(pid) or pid)
        return pgids

    def _pick_coresident_gpu(self, model_name: str) -> int:
        """GPU whose free VRAM covers this model's min-viable footprint.

        The budget is the GPU's *current* free VRAM (nvidia-smi) minus what
        co-resident models still starting there will take, against the model's
        min-viable util plus HARD_MARGIN_MIB — the same test the backend applies
        at spawn.  A GPU without a primary wins over one with more free VRAM, so
        a primary's replica keeps its room.  Raises GPUBusyError if none fits."""
        config = self.model_configs[model_name]
        allowed = config.allowed_gpu_ids
        gpus = sorted({
            s.gpu_id for s in self.slots if allowed is None or s.gpu_id in allowed
        })
        min_util = MODEL_MIN_GPU_MEM_UTIL.get(model_name, GPU_MEM_UTIL_FLOOR)
        options: list[tuple[bool, float, int]] = []
        report: list[str] = []
        need_gib = 0.0
        for gpu in gpus:
            total, free = _gpu_total_mib(gpu), _gpu_free_mib(gpu)
            if not total or free is None:
                report.append(f"GPU {gpu}: unreadable")
                continue
            reserved = sum(
                MODEL_GPU_MEM_UTIL.get(
                    b.model_name,
                    MODEL_MIN_GPU_MEM_UTIL.get(b.model_name, GPU_MEM_UTIL_FLOOR),
                ) * total
                for b in self._coresident_backends_on(gpu) if not b._ready
            )
            avail = free - reserved
            need = min_util * total + HARD_MARGIN_MIB
            need_gib = need / 1024.0
            report.append(f"GPU {gpu}: {avail / 1024.0:.1f} GiB free")
            if avail >= need:
                has_primary = any(
                    s.gpu_id == gpu and s.backend is not None for s in self.slots
                )
                options.append((has_primary, -avail, gpu))
        if not options:
            raise GPUBusyError(
                f"No GPU has room for '{model_name}': it needs >={need_gib:.1f} GiB "
                f"free (min-viable util {min_util}); " + "; ".join(report) + ". "
                f"Retry once a GPU frees up."
            )
        return min(options)[2]

    async def _claim_coresident_locked(self, model_name: str) -> GpuBackend:
        """Claim a lane on the GPU chosen by budget.  Caller holds _router_lock."""
        gpu = self._pick_coresident_gpu(model_name)
        lane = self._lane_slot(model_name, gpu)
        b = self._make_backend(model_name, lane)
        lane.backend = b
        await b.start()
        return b

    async def reclaim_coresident_for(self, primary: GpuBackend) -> None:
        """Primaries outrank co-resident models for VRAM.  If the primary about
        to spawn on this GPU would not fit at its min-viable util, stop idle
        co-resident models there until it does.  Busy ones are left running (the
        spawn then fails as "too tight", as it would for any neighbour)."""
        gpu = primary.gpu_id
        victims = [
            b for b in self._coresident_backends_on(gpu)
            if b._ready and b.is_running and b._active_requests == 0
        ]
        if not victims:
            return
        total, free = _gpu_total_mib(gpu), _gpu_free_mib(gpu)
        if not total or free is None:
            return
        need = (
            MODEL_MIN_GPU_MEM_UTIL.get(primary.model_name, GPU_MEM_UTIL_FLOOR) * total
            + HARD_MARGIN_MIB
        )
        for victim in victims:
            if free >= need:
                return
            self.log.warning(
                f"Reclaiming VRAM on GPU {gpu} for {primary.model_name}: stopping idle "
                f"co-resident {victim.model_name} (free {free / 1024.0:.1f} GiB, "
                f"need {need / 1024.0:.1f} GiB)"
            )
            await victim.stop()
            # Let the killed process group release its CUDA memory before re-reading.
            await asyncio.sleep(3)
            free = _gpu_free_mib(gpu)
            if free is None:
                return

    def _running_backends(self, model_name: str) -> list[GpuBackend]:
        """Backends that are fully ready to serve (spawned AND healthy)."""
        return [
            s.backend for s in self._all_slots()
            if s.backend
            and s.backend.model_name == model_name
            and s.backend._ready
            and s.backend.is_running
        ]

    def _claimed_backends(self, model_name: str) -> list[GpuBackend]:
        """Running or mid-spawn, excluding permanently-failed ones."""
        return [
            s.backend for s in self._all_slots()
            if s.backend
            and s.backend.model_name == model_name
            and not s.backend._failed
        ]

    def _free_slots(self) -> list[GpuSlot]:
        """Free slots, also cleaning up any dead-process stale entries."""
        result = []
        for s in self.slots:
            if s.backend is None:
                result.append(s)
            elif s.backend._failed:
                # Permanently failed backend — free the slot
                s.backend = None
                result.append(s)
        return result

    @staticmethod
    def _by_free_vram(slots: "list[GpuSlot]") -> "list[GpuSlot]":
        """Slots ordered emptiest-GPU-first, so a spawn lands where it fits.

        Picking slots in declaration order meant always trying GPU 1 first, and
        GPU 1 shares the card with video-transcribe-service (~1.6 GiB).  That
        leaves ~27.9 GiB free — enough that the card looks usable, but 0.8 GiB
        short of the 28.7 GiB a 35B needs at min-viable util, so the spawn was
        refused with "too tight" while GPU 0 sat completely empty.  That is
        every one of the 66 such failures in the current log: 66 on GPU 1, none
        on GPU 0.

        nvidia-smi is queried per slot here; the list is at most a handful of
        entries and this runs only on the spawn path, not per request.  Slots
        whose GPU cannot be read sort last rather than blocking selection."""
        return sorted(
            slots,
            key=lambda s: _gpu_free_mib(s.gpu_id) or -1.0,
            reverse=True,
        )

    def _allowed_gpus(self, model_name: str) -> "set[int] | None":
        """GPU IDs this model may occupy, or None if unconstrained.

        A None return means the model can run on any GPU in the slot pool.
        """
        cfg = self.model_configs.get(model_name)
        return cfg.allowed_gpu_ids if cfg is not None else None

    # ── Admin / dashboard ──────────────────────────────────────────────────────

    def status(self) -> dict:
        """Live snapshot of every slot for the /admin dashboard."""
        now = time.monotonic()
        slots = []
        for s in self.slots:
            b = s.backend
            allowed_here = {
                m for m in self.model_configs
                if (g := self._allowed_gpus(m)) is None or s.gpu_id in g
            }
            entry = {
                "slot_id": s.slot_id,
                "gpu_id":  s.gpu_id,
                "port":    s.port,
                "allowed_models": sorted(allowed_here),
                "msg":     self._admin_msg.get(s.slot_id, ""),
            }
            if b is None:
                entry.update(model=None, state="free", ready=False,
                             running=False, active_requests=0, idle_seconds=None)
            else:
                entry.update(
                    model=b.model_name,
                    ready=b._ready,
                    running=b.is_running,
                    failed=b._failed,
                    active_requests=b._active_requests,
                    # last_activity is stamped at request start/end, so during
                    # a long request "now - last_activity" grows while the
                    # backend is busy.  A busy backend is not idle — report
                    # None (rendered as "—") whenever requests are in flight.
                    idle_seconds=(None if b._active_requests > 0
                                  else int(now - b.last_activity)),
                    adopted=b._adopted_pid is not None,
                    state=("failed" if b._failed else
                           "ready" if (b._ready and b.is_running) else
                           "starting"),
                )
            slots.append(entry)
        coresident = []
        for lane in self._lanes.values():
            b = lane.backend
            if b is None:
                continue
            coresident.append({
                "slot_id": lane.slot_id,
                "gpu_id": lane.gpu_id,
                "port": lane.port,
                "model": b.model_name,
                "ready": b._ready,
                "running": b.is_running,
                "failed": b._failed,
                "active_requests": b._active_requests,
                "idle_seconds": (None if b._active_requests > 0
                                 else int(now - b.last_activity)),
                "idle_timeout": b._base_idle_timeout(),
                "adopted": b._adopted_pid is not None,
                "state": ("failed" if b._failed else
                          "ready" if (b._ready and b.is_running) else
                          "starting"),
            })
        return {
            "slots": slots,
            "coresident": coresident,
            "models": list(self.model_configs),
            "model_limits": {
                config.served_name: {"max_num_seqs": config.max_num_seqs}
                for config in self.model_configs.values()
                if config.max_num_seqs is not None
            },
            "idle_timeout": IDLE_TIMEOUT,
            "wake_timeout": WAKE_TIMEOUT,
            # Seconds since this model last had a queued request on ANY of its
            # instances (null = none seen since start).  This is the signal the
            # replica scale-in guard reads, so surfacing it here is what makes
            # a "should that replica have been reclaimed?" question answerable
            # after the fact instead of a guess.
            "quiet_for": {
                model_name: round(now - stamp, 1)
                for model_name, stamp in self._last_backlog_at.items()
            },
            "replica_quiet_before_reclaim": REPLICA_QUIET_BEFORE_RECLAIM,
        }

    def _track_admin(self, coro) -> None:
        """Run an admin coroutine in the background, keeping a reference."""
        task = asyncio.create_task(coro)
        self._admin_tasks.add(task)
        task.add_done_callback(self._admin_tasks.discard)

    def _slot_by_id(self, slot_id: int) -> GpuSlot:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        raise KeyError(f"no slot with id {slot_id}")

    async def admin_kill(self, slot_id: int) -> None:
        """Unload whatever vLLM occupies a slot, freeing it."""
        slot = self._slot_by_id(slot_id)
        b = slot.backend
        if b is None:
            self._admin_msg[slot_id] = "already free"
            return
        self._admin_msg[slot_id] = f"killing {b.model_name}…"
        try:
            await b.stop()
            self._admin_msg[slot_id] = "freed"
        except Exception as exc:
            self._admin_msg[slot_id] = f"kill failed: {exc}"

    async def admin_start(self, slot_id: int, model_name: str) -> None:
        """Spawn a specific model on a specific (currently free) slot."""
        if model_name not in self.model_configs:
            self._admin_msg[slot_id] = f"unknown model {model_name}"
            return
        async with self._router_lock:
            slot = self._slot_by_id(slot_id)
            if slot.backend is not None:
                self._admin_msg[slot_id] = (
                    f"slot busy ({slot.backend.model_name}) — kill or switch first")
                return
            allowed = self._allowed_gpus(model_name)
            if allowed is not None and slot.gpu_id not in allowed:
                self._admin_msg[slot_id] = (
                    f"{model_name} not allowed on GPU {slot.gpu_id}")
                return
            if self.model_configs[model_name].coresident:
                self._admin_msg[slot_id] = (
                    f"{model_name} is co-resident: it starts on demand by VRAM "
                    f"budget, not into a slot")
                return
            b = self._make_backend(model_name, slot)
            slot.backend = b           # CLAIM
            await b.start()            # init session + watchdog
        self._admin_msg[slot_id] = f"starting {model_name}…"
        try:
            await b._ensure_running()
            self._admin_msg[slot_id] = f"{model_name} ready"
        except Exception as exc:
            if b.slot.backend is b:
                b.slot.backend = None
            self._admin_msg[slot_id] = f"start failed: {exc}"

    async def admin_switch(self, slot_id: int, model_name: str) -> None:
        """Kill the current backend on a slot then start a different model there."""
        await self.admin_kill(slot_id)
        await self.admin_start(slot_id, model_name)

    # ── Core: get or start a backend ──────────────────────────────────────────

    async def _get_or_start(self, model_name: str) -> list[GpuBackend]:
        """Return ≥1 ready GpuBackend instances, starting one on a free slot if needed.

        Raises GPUBusyError if all slots are occupied by other models.
        Raises RuntimeError if vLLM fails to start.
        """
        # ① Fast path — model already running
        running = self._running_backends(model_name)
        if running:
            return running

        # ② Model is mid-spawn by a concurrent request — wait for it
        claimed = self._claimed_backends(model_name)
        if claimed:
            b = claimed[0]
            await b._ensure_running()
            running = self._running_backends(model_name)
            if running:
                return running
            raise RuntimeError(f"Startup failed for '{model_name}' — see {b.log_path}.")

        # ③ Need to claim a slot — serialise with the router lock
        async with self._router_lock:
            # Re-check after acquiring the lock
            running = self._running_backends(model_name)
            if running:
                return running

            claimed = self._claimed_backends(model_name)
            if claimed:
                b = claimed[0]   # another coroutine claimed while we waited — fall through
            elif self.model_configs[model_name].coresident:
                b = await self._claim_coresident_locked(model_name)
            else:
                free = self._free_slots()
                # Apply GPU affinity: drop slots on GPUs this model cannot use.
                allowed = self._allowed_gpus(model_name)
                compatible = [s for s in free if allowed is None or s.gpu_id in allowed]
                if not compatible:
                    occupied = [(s.slot_id, s.current_model) for s in self.slots
                                if not s.is_free]
                    if allowed is not None and free:
                        # Free slots exist but none are on an allowed GPU.
                        raise GPUBusyError(
                            f"No compatible GPU slot for '{model_name}' "
                            f"(allowed GPUs: {allowed}; free slots are on "
                            f"non-allowed GPUs). "
                            f"Retry after {IDLE_TIMEOUT}s idle."
                        )
                    raise GPUBusyError(
                        f"All GPU slots are occupied: {occupied}. "
                        f"Retry after {IDLE_TIMEOUT}s idle."
                    )
                slot   = self._by_free_vram(compatible)[0]
                b      = self._make_backend(model_name, slot)
                slot.backend = b          # CLAIM — blocks other models from this slot
                await b.start()           # init session + idle watchdog

        # ④ Spawn OUTSIDE the lock (takes up to WAKE_TIMEOUT seconds)
        try:
            await b._ensure_running()
        except Exception:
            # Release slot so other models can use it
            if b.slot.backend is b:
                b.slot.backend = None
            raise

        running = self._running_backends(model_name)
        return running if running else [b]

    # ── Scale-out ──────────────────────────────────────────────────────────────

    def _log_scale_decline(self, model_name: str, reason: str) -> None:
        """Log why a scale-out attempt declined, at most once per 300s per
        (model, reason).  The saturation loop re-attempts every ~SUSTAIN seconds
        while a model stays saturated; without dedup a persistent blocker (GPU
        busy, insufficient VRAM) would spam an identical line every cycle and
        make it look like scale-out is firing when it's actually stuck."""
        prev = self._scale_decline_log.get(model_name)
        now = time.monotonic()
        if prev is not None and prev[0] == reason and now - prev[1] < 300:
            return
        self._scale_decline_log[model_name] = (reason, now)
        self.log.info(f"Scale-out for {model_name} declined: {reason}")

    async def _maybe_scale_out(self, model_name: str) -> None:
        """If all instances are saturated and a slot is available, spawn another.

        "Available" means either a truly free slot (no backend) OR a slot whose
        backend is a *different* model that is currently idle (0 active requests).
        In the latter case we evict the idle model first so the saturated model
        can use the slot.  This is important in a 2-slot system where both slots
        are always occupied by different models — without this, scale-out would
        never trigger even when one GPU is at 100% and the other is fully idle.
        """
        if self.model_configs[model_name].coresident:
            return   # one small instance per GPU by budget; never a slot replica

        # Respect cooldown after a previous failure (prevents crash-loop when a
        # slot cannot physically start the model, e.g. insufficient free VRAM).
        last_fail = self._scale_fail_time.get(model_name, 0)
        cooldown_left = self.SCALE_OUT_COOLDOWN - (time.monotonic() - last_fail)
        if cooldown_left > 0:
            self._log_scale_decline(
                model_name,
                f"in cooldown after a prior failed spawn ({int(cooldown_left)}s left)",
            )
            return

        running = self._running_backends(model_name)
        if not running:
            return
        if len(running) > 1:
            return   # already scaled out onto multiple slots

        slot:    GpuSlot     | None = None
        new_b:   GpuBackend  | None = None
        evict_b: GpuBackend  | None = None   # idle foreign backend to evict

        async with self._router_lock:
            running = self._running_backends(model_name)
            if not running:
                return
            if len(running) > 1:
                return   # another scale-out already completed

            running_slot_ids = {b.slot.slot_id for b in running}
            allowed = self._allowed_gpus(model_name)
            free = [
                s for s in self._free_slots()
                if s.slot_id not in running_slot_ids
                and (allowed is None or s.gpu_id in allowed)
            ]

            # Apply minimum-free-memory filter to truly free slots too.
            # A truly free slot on a GPU that doesn't have enough room for this
            # model is just as doomed as an eviction target — skip it early.
            min_free_gib = MODEL_MIN_FREE_GIB.get(model_name)
            if min_free_gib is not None and free:
                def _slot_has_room(s: GpuSlot) -> bool:
                    free_mib = _gpu_free_mib(s.gpu_id)
                    if free_mib is None:
                        return True   # can't check, optimistically allow
                    # For truly free slots the vLLM memory is 0 (no vLLM yet).
                    vllm_used = _gpu_vllm_used_mib(s.gpu_id)
                    return (free_mib + vllm_used) / 1024.0 >= min_free_gib
                viable = [s for s in free if _slot_has_room(s)]
                if not viable:
                    self._log_scale_decline(
                        model_name,
                        f"free slot(s) lack VRAM (need {min_free_gib:.1f} GiB; "
                        f"another process likely holds the GPU)",
                    )
                    return
                free = viable

            if not free:
                # No truly free slot — look for a slot whose backend is a
                # *different* idle model we can evict to make room.
                evictable = [
                    s for s in self.slots
                    if s.slot_id not in running_slot_ids
                    and s.backend is not None
                    and not s.backend._failed
                    and s.backend.model_name != model_name
                    and s.backend._active_requests == 0
                    and s.backend._ready              # never evict a mid-spawn backend
                    and (allowed is None or s.gpu_id in allowed)
                ]
                if not evictable:
                    self._log_scale_decline(
                        model_name,
                        "no free slot and no idle foreign model to evict",
                    )
                    return

                # Pre-eviction memory check: estimate free GPU memory after
                # eviction by adding the victim's vLLM memory to the current
                # free.  Avoids destructively clearing a slot only to
                # immediately fail the spawn due to insufficient VRAM.
                min_free_gib = MODEL_MIN_FREE_GIB.get(model_name)
                if min_free_gib is not None:
                    valid_targets = []
                    for s in evictable:
                        free_mib = _gpu_free_mib(s.gpu_id)
                        if free_mib is None:
                            valid_targets.append(s)   # can't query, allow it
                            continue
                        # All vLLM processes on this GPU will be gone after eviction.
                        vllm_used_mib = _gpu_vllm_used_mib(s.gpu_id)
                        free_after_gib = (free_mib + vllm_used_mib) / 1024.0
                        if free_after_gib >= min_free_gib:
                            valid_targets.append(s)
                        else:
                            self.log.info(
                                f"Scale-out for {model_name}: skipping eviction of "
                                f"{s.backend.model_name} on slot {s.slot_id} "
                                f"(GPU {s.gpu_id}) — only {free_after_gib:.1f} GiB "
                                f"would be free after eviction, need {min_free_gib:.1f} GiB"
                            )
                    evictable = valid_targets

                if not evictable:
                    self._log_scale_decline(
                        model_name,
                        f"idle model(s) found to evict, but GPU would still lack "
                        f"{min_free_gib:.1f} GiB after eviction (another process on "
                        f"the GPU)",
                    )
                    return

                victim_slot = evictable[0]
                evict_b = victim_slot.backend   # save ref before we overwrite
                # Atomically claim the slot — prevents any other request from
                # grabbing it while we're killing the incumbent vLLM.
                victim_slot.backend = None      # detach old backend
                free = [victim_slot]

            slot   = self._by_free_vram(free)[0]
            new_b  = self._make_backend(model_name, slot)
            slot.backend = new_b
            await new_b.start()

        # ── Outside the router lock ─────────────────────────────────────────────
        # If we evicted a foreign backend, kill its vLLM process first so its GPU
        # memory is freed before we try to spawn on the same GPU.
        if evict_b is not None:
            self.log.info(
                f"Scale-out for {model_name}: evicting idle {evict_b.model_name} "
                f"from slot {slot.slot_id} (GPU {slot.gpu_id})"
            )
            async with evict_b._lock:
                await evict_b._kill_process_locked()

        self.log.info(
            f"Scale-out: spawning {model_name} on slot {slot.slot_id} (GPU {slot.gpu_id})"
        )
        try:
            await new_b._ensure_running()
            active_slots = [s.slot_id for s in self.slots if s.current_model == model_name]
            self.log.info(f"Scale-out complete: {model_name} now on slots {active_slots}")
            # Clear cooldown + decline-dedup on success so future cycles log fresh.
            self._scale_fail_time.pop(model_name, None)
            self._scale_decline_log.pop(model_name, None)
        except Exception as exc:
            self.log.warning(
                f"Scale-out failed for {model_name} on slot {slot.slot_id}: {exc} "
                f"— cooling down for {self.SCALE_OUT_COOLDOWN}s"
            )
            self._scale_fail_time[model_name] = time.monotonic()
            if slot.backend is new_b:
                slot.backend = None

    def _pick(self, backends: list[GpuBackend]) -> GpuBackend:
        """Least-connections: route to the backend with fewest in-flight requests.

        Round-robin doesn't account for queue depth — a backend that started
        earlier can accumulate a large backlog while a newer one sits idle.
        Least-connections naturally drains the lighter backend first.
        Ties are broken by insertion order (oldest backend first).
        """
        return min(backends, key=lambda b: b._active_requests)

    def _trigger_scale_out(self, model_name: str) -> None:
        task = asyncio.create_task(self._maybe_scale_out(model_name))
        self._scale_tasks.add(task)
        task.add_done_callback(self._scale_tasks.discard)

    def start_saturation_monitor(self) -> None:
        """Launch the background queue sampler that drives sustained scale-out."""
        if self._sat_task is None or self._sat_task.done():
            self._sat_task = asyncio.create_task(self._saturation_loop())

    async def _saturation_loop(self) -> None:
        """Sample each single-slot model's real vLLM queue (num_requests_waiting,
        summed) every 10s and drive a sliding-window backlog accumulator P per
        model — the sum of per-sample accruals from the last SCALE_WINDOW seconds:

            waiting >= 2 → record (now, dt / sustain(waiting))   # deeper = faster
            waiting <= 1 → record nothing (old samples age out on their own)
            P >= 1.0     → attempt scale-out, clear the window

        Time at a shallow depth counts proportionally toward a deeper threshold
        (waiting=2's 1/300-per-sec is 0.66× waiting=3's 1/200), and an oscillating
        backlog accumulates across drain/refill episodes inside the window instead
        of resetting each time the queue momentarily empties.  Models already on
        >1 slot, or with nothing running, are skipped.  Decided here on sustained
        real backlog, never on a transient in-flight spike."""
        SAMPLE = 10
        try:
            while True:
                await asyncio.sleep(SAMPLE)
                now = time.monotonic()
                # Real elapsed since last sample (cap against loop stalls so a
                # hiccup can't over-credit a model toward scale-out).
                dt = (now - self._sat_last_tick) if self._sat_last_tick else 0.0
                dt = min(dt, SAMPLE * 3)
                self._sat_last_tick = now

                for model_name in self.model_configs:
                    if self.model_configs[model_name].coresident:
                        continue
                    running = self._running_backends(model_name)
                    if not running:
                        self._scale_accrual.pop(model_name, None)
                        self._last_backlog_at.pop(model_name, None)
                        continue
                    try:
                        depths = await asyncio.gather(
                            *(b.queue_depth() for b in running)
                        )
                    except Exception:
                        continue
                    waiting = sum(depths)
                    # Stamp model-wide backlog on EVERY sample, including while
                    # scaled out — scale-in reads this to avoid reclaiming a
                    # replica during a lull between bursts.
                    if waiting > 0:
                        self._last_backlog_at[model_name] = now

                    if len(running) != 1:
                        # Already scaled out: no accrual (the backlog stamp
                        # above is still what scale-in needs).
                        self._scale_accrual.pop(model_name, None)
                        continue

                    window = self._scale_accrual.get(model_name)
                    had_progress = bool(window)
                    if window:
                        cutoff = now - SCALE_WINDOW
                        while window and window[0][0] < cutoff:
                            window.popleft()

                    sustain = _scale_sustain_for(waiting)
                    if sustain is not None and dt > 0:
                        if window is None:
                            window = self._scale_accrual[model_name] = deque()
                        window.append((now, dt / sustain))
                        if not had_progress:
                            self.log.info(
                                f"{model_name}: waiting={waiting} backlog — scale-out "
                                f"accrual started (tier {int(sustain)}s, "
                                f"window {int(SCALE_WINDOW)}s)"
                            )
                    elif had_progress and not window:
                        # Last accrual aged past the window with no new backlog.
                        self._scale_accrual.pop(model_name, None)
                        self.log.info(
                            f"{model_name}: backlog window drained — accumulator empty"
                        )
                        continue

                    P = sum(inc for _, inc in window) if window else 0.0
                    if P >= 1.0:
                        self.log.info(
                            f"{model_name}: waiting={waiting} backlog sustained "
                            f"(windowed P≥1.0 over {int(SCALE_WINDOW)}s) — "
                            f"attempting scale-out"
                        )
                        self._trigger_scale_out(model_name)
                        self._scale_accrual.pop(model_name, None)   # start fresh
        except asyncio.CancelledError:
            pass

    # ── aiohttp request handler ────────────────────────────────────────────────

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if request.method == "GET" and request.path in ("/health", "/v1/health"):
            return web.Response(status=200, text="OK")

        if request.method == "GET" and request.path in ("/v1/models", "/models"):
            return web.json_response({
                "object": "list",
                "data": [
                    {"id": name, "object": "model", "owned_by": "local"}
                    for name in self.model_configs
                ],
            })

        # ── Admin dashboard ─────────────────────────────────────────────────────
        # Bare root and favicon are not OpenAI routes; send a browser to /admin
        # instead of falling through to the "cannot determine model" error.
        if request.method == "GET" and request.path in ("/", "/favicon.ico"):
            return web.HTTPFound("/admin")
        if request.method == "GET" and request.path in ("/admin", "/admin/"):
            return web.Response(text=ADMIN_HTML, content_type="text/html")
        if request.method == "GET" and request.path == "/admin/status":
            return web.json_response(self.status())
        # Read-only public status (see STATUS_HTML comment).  GET only; the
        # tunnel's path route (^/status) can reach nothing but these two.
        if request.method == "GET" and request.path in ("/status", "/status/"):
            return web.Response(text=STATUS_HTML, content_type="text/html")
        if request.method == "GET" and request.path == "/status.json":
            return web.json_response(self.status())
        if request.method == "POST" and request.path in (
            "/admin/kill", "/admin/start", "/admin/switch"
        ):
            try:
                data = await request.json()
                slot_id = int(data["slot_id"])
            except Exception as exc:
                return web.json_response(
                    {"error": f"bad request: {exc}"}, status=400)
            action = request.path.rsplit("/", 1)[1]
            if action == "kill":
                self._track_admin(self.admin_kill(slot_id))
            else:
                model = data.get("model")
                if not model:
                    return web.json_response(
                        {"error": "missing 'model'"}, status=400)
                fn = self.admin_start if action == "start" else self.admin_switch
                self._track_admin(fn(slot_id, model))
            return web.json_response({"ok": True, "action": action, "slot_id": slot_id})

        body = await request.read()
        model_name = self._extract_model(body, request.headers.get("Content-Type", ""))
        if not model_name:
            return web.Response(
                status=400, content_type="application/json",
                body=json.dumps({"error": {
                    "message": "Cannot determine model from request body",
                    "type": "invalid_request_error",
                }}),
            )

        if model_name not in self.model_configs:
            self.log.warning(f"Unknown model '{model_name}'")
            return web.Response(
                status=404, content_type="application/json",
                body=json.dumps({"error": {
                    "message": f"Unknown model: {model_name}",
                    "type": "invalid_request_error",
                }}),
            )

        try:
            parsed_body = json.loads(body)
        except (json.JSONDecodeError, TypeError, ValueError):
            parsed_body = None
        config = self.model_configs[model_name]
        invalid = self._validate_model_request(request, parsed_body, config)
        if invalid is not None:
            return invalid
        if config.request_kind == "speech":
            # The public speech contract is deliberately MP3-only.  vLLM-Omni
            # can encode it natively, and LiteLLM 1.86.2 labels every speech
            # response as audio/mpeg.  Normalising an omitted/case-varied value
            # here keeps the bytes, filename convention, and Content-Type in
            # agreement without adding an ffmpeg transcode layer.
            parsed_body["response_format"] = "mp3"
            body = json.dumps(parsed_body, ensure_ascii=False).encode("utf-8")

        # Probe / latency-check detection (e.g. LiteLLM latency-based-routing).
        # Rule: only test models that are already loaded.
        #   • Model loaded  → let the request through for real latency measurement.
        #   • Model cold    → return 503 immediately, no spawn triggered.
        #     LiteLLM treats the 503 as "high latency / unavailable" and avoids
        #     routing to this model until it comes up naturally via a real request.
        try:
            is_probe = (
                config.request_kind == "chat"
                and isinstance(parsed_body, dict)
                and parsed_body.get("max_tokens", 9999) <= 1
                and not parsed_body.get("messages", [{}])[-1].get("content", "").strip()
            )
            if is_probe:
                running = self._running_backends(model_name)
                if not running:
                    # Model is cold — reject probe without spawning.
                    return web.Response(
                        status=503, content_type="application/json",
                        body=json.dumps({"error": {
                            "message": (
                                f"Model '{model_name}' is not currently loaded. "
                                "Probe rejected to prevent cold start."
                            ),
                            "type": "service_unavailable",
                        }}),
                    )
                # Model is warm — fall through and measure real latency.
        except Exception:
            pass

        # Sticky routing metadata is backend-independent — parse it once, before
        # the retry loop below.  x-task-id pins all turns of one task to the same
        # vLLM slot for prefix cache reuse; falls back to least-connections.
        parsed_body_for_sticky = parsed_body if isinstance(parsed_body, dict) else None
        if config.request_kind == "transcription":
            sticky_slot = None
            task_id = None
            msgs = []
            approx_chars = len(body)      # upload size, for the routing log
        elif config.request_kind == "speech":
            sticky_slot = None
            task_id = None
            msgs = []
            approx_chars = len(parsed_body_for_sticky["input"])
        else:
            sticky_slot, _sticky_reason = _sticky_slot_for(
                parsed_body_for_sticky, request.headers
            )
            task_id = _extract_task_id(parsed_body_for_sticky, request.headers)
            # Approximate prompt size (sum of message-content chars) so the
            # pipeline team can grep-by-task and see whether history is growing.
            msgs = (parsed_body_for_sticky or {}).get("messages", []) or []
            approx_chars = sum(
                len(m.get("content", "")) if isinstance(m.get("content"), str)
                else sum(
                    len(p.get("text", "") or "")
                    for p in (m.get("content") or []) if isinstance(p, dict)
                )
                for m in msgs if isinstance(m, dict)
            )

        # A backend can vanish BEFORE it streams a single byte — a crash-recycle
        # SIGKILL or a reload racing this in-flight request.  That surfaces as
        # BackendGoneError (raised only pre-stream, never mid-body).  Since
        # nothing reached the client, respawn and retry ONCE; a would-be 502 thus
        # becomes a 200, or at worst a retryable 503.  This is what keeps a
        # reload / cold-start from ever surfacing a 502 at the main address.
        last_gone: BackendGoneError | None = None
        for attempt in range(2):
            try:
                backends = await self._get_or_start(model_name)
            except GPUBusyError as exc:
                self.log.warning(str(exc))
                return web.Response(
                    status=503, content_type="application/json",
                    body=json.dumps({"error": {"message": str(exc), "type": "gpu_busy"}}),
                )
            except RuntimeError as exc:
                self.log.error(str(exc))
                return web.Response(
                    status=503, content_type="application/json",
                    body=json.dumps({"error": {"message": str(exc), "type": "startup_failed"}}),
                )

            sticky_backend: GpuBackend | None = None
            if sticky_slot is not None:
                sticky_backend = next(
                    (b for b in backends if b.slot.slot_id == sticky_slot), None
                )
            if sticky_backend is not None:
                backend = sticky_backend
            else:
                backend = self._pick(backends)
                # Scale-out is decided by the background saturation monitor based
                # on vLLM's sustained real queue — not triggered per request here.

            # Record affinity + log routing (only the first attempt for the
            # no-task path, to avoid duplicate lines on a transparent retry).
            if config.request_kind == "speech" and attempt == 0:
                instructions = parsed_body_for_sticky.get("instructions")
                instructions_chars = (
                    len(instructions) if isinstance(instructions, str) else 0
                )
                self.log.info(
                    f"Speech request: model={model_name} → slot "
                    f"{backend.slot.slot_id} (input_chars={approx_chars}, "
                    f"instructions_chars={instructions_chars})"
                )
            elif config.request_kind == "transcription" and attempt == 0:
                self.log.info(
                    f"Transcription request: model={model_name} → lane slot "
                    f"{backend.slot.slot_id} (GPU {getattr(backend, 'gpu_id', '?')}, "
                    f"upload_bytes={approx_chars})"
                )
            elif task_id:
                hit_status = "hit" if sticky_backend is not None else "fresh"
                retry_note = "" if attempt == 0 else f" (retry {attempt})"
                self.log.info(
                    f"Sticky chat/completions: task_id={task_id} {hit_status} → slot {backend.slot.slot_id} "
                    f"(msgs={len(msgs)}, chars≈{approx_chars}){retry_note}"
                )
                _set_task_affinity(task_id, backend.slot.slot_id)
            elif attempt == 0:
                hdr_summary = ", ".join(
                    f"{k.lower()}" for k in request.headers
                    if k.lower().startswith(("x-", "anthropic-", "authorization", "user-agent"))
                )
                body_keys = sorted(parsed_body_for_sticky.keys()) if parsed_body_for_sticky else []
                self.log.info(
                    f"chat/completions no task_id (headers: [{hdr_summary}], body keys: {body_keys}, "
                    f"msgs={len(msgs)}, chars≈{approx_chars})"
                )

            try:
                return await backend.proxy(request, body)
            except BackendGoneError as exc:
                last_gone = exc
                self.log.warning(
                    f"Backend on slot {backend.slot.slot_id} gone before response "
                    f"(attempt {attempt + 1}/2) for {model_name}: {exc} — "
                    f"{'retrying' if attempt == 0 else 'giving up'}"
                )

        # Both attempts saw the backend disappear pre-response.  Return a
        # retryable 503 — never a 502 — since the request produced no output.
        self.log.error(f"Backend unavailable after retry for {model_name}: {last_gone}")
        return web.Response(
            status=503, content_type="application/json",
            body=json.dumps({"error": {
                "message": f"Backend for '{model_name}' was momentarily unavailable; please retry.",
                "type": "service_unavailable",
            }}),
        )

    @staticmethod
    def _extract_model(body: bytes, content_type: str = "") -> str | None:
        """The routing model: the JSON `model` key, or for a multipart upload
        (audio transcription) the `model` form field."""
        if not body:
            return None
        if content_type.lower().startswith("multipart/form-data"):
            return DynamicRouter._multipart_field(body, content_type, "model")
        try:
            return json.loads(body).get("model")
        except (ValueError, AttributeError):
            return None

    @staticmethod
    def _multipart_field(body: bytes, content_type: str, name: str) -> str | None:
        """Text value of one multipart form field, without copying the body.

        Only each part's header block (up to its first blank line) is searched
        for the field name, so audio bytes that happen to contain form-data
        text cannot be mistaken for a field."""
        match = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type, re.I)
        if not match:
            return None
        delim = b"--" + (match.group(1) or match.group(2)).encode()
        needle = f'name="{name}"'.encode()
        pos = body.find(delim)
        while pos != -1:
            start = pos + len(delim)
            if body[start:start + 2] == b"--":
                return None                      # closing delimiter
            nxt = body.find(delim, start)
            end = nxt if nxt != -1 else len(body)
            head_end = body.find(b"\r\n\r\n", start, end)
            if head_end != -1 and needle in body[start:head_end]:
                value = body[head_end + 4:end]
                if value.endswith(b"\r\n"):
                    value = value[:-2]
                return value.decode("utf-8", "replace").strip() or None
            pos = nxt
        return None

    @staticmethod
    def _invalid_request(message: str, param: str | None = None) -> web.Response:
        error = {"message": message, "type": "invalid_request_error"}
        if param is not None:
            error["param"] = param
        return web.Response(
            status=400,
            content_type="application/json",
            body=json.dumps({"error": error}),
        )

    def _validate_model_request(
        self,
        request: web.Request,
        parsed_body: object,
        config: ModelConfig,
    ) -> web.Response | None:
        if config.request_kind == "transcription":
            if request.method != "POST" or request.path != "/v1/audio/transcriptions":
                return self._invalid_request(
                    "Transcription models are only available through "
                    "POST /v1/audio/transcriptions"
                )
            return None
        if config.request_kind != "speech":
            return None
        if request.method != "POST" or request.path != "/v1/audio/speech":
            return self._invalid_request(
                "Speech models are only available through POST /v1/audio/speech"
            )
        if not isinstance(parsed_body, dict):
            return self._invalid_request("Request body must be a JSON object")
        input_text = parsed_body.get("input")
        if not isinstance(input_text, str) or not input_text.strip():
            return self._invalid_request(
                "'input' must be a non-empty string", param="input"
            )
        if (
            config.max_input_chars is not None
            and len(input_text) > config.max_input_chars
        ):
            return self._invalid_request(
                f"'input' exceeds the {config.max_input_chars} character limit",
                param="input",
            )
        return None

    # ── Adoption ───────────────────────────────────────────────────────────────

    ADOPT_BOOT_WAIT = 5     # max seconds to wait for a booting vLLM to expose /v1/models
    # NOTE: Keep this short (≤5s). Long values cause instance pile-up: with RestartSec=10
    # and ADOPT_BOOT_WAIT=120, up to 12 instances accumulate waiting for the same vLLM.
    # When vLLM finally responds, all instances complete adoption, port 8002 contention
    # causes crashes, and the ExecStartPre fuser (now removed) SIGKILLs stable instances.
    # At 5s, if vLLM isn't ready yet, skip adoption — first request triggers a fresh spawn.

    async def adopt_existing_backends(self) -> None:
        """Probe each slot for a vLLM process and adopt it.

        Paired with the systemd unit's KillMode=process so model_manager
        restarts don't kill vLLM.  Detects vLLM via pgrep so we recognise
        instances that are still cold-starting (port not yet bound).  Runs
        all slots concurrently — at worst one slot delays startup by
        ADOPT_BOOT_WAIT seconds (instead of N × slots).
        """
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5),
        ) as session:
            await asyncio.gather(*(
                self._try_adopt_slot(slot, session) for slot in self.slots
            ))
            # Co-resident vLLM instances survive a manager restart (KillMode=
            # process) on their deterministic lane ports; adopt any that are up.
            lanes = [
                self._lane_slot(model, gpu)
                for gpu in sorted({s.gpu_id for s in self.slots})
                for model in self._coresident_names()
                if _find_vllm_pid_for_port(self._lane_port(model, gpu)) is not None
            ]
            await asyncio.gather(*(
                self._try_adopt_slot(lane, session) for lane in lanes
            ))

    async def _try_adopt_slot(self, slot: GpuSlot,
                              session: aiohttp.ClientSession) -> None:
        pid = _find_vllm_pid_for_port(slot.port)
        if pid is None:
            return   # slot is genuinely free
        # Identify the model from the process cmdline (avoids needing the
        # vLLM api-key to call /v1/models).
        served_name = _read_served_model_name(pid)
        if not served_name:
            self.log.warning(
                f"Slot {slot.slot_id}: pid {pid} cmdline lacks --served-model-name — skipping"
            )
            return
        match = next(
            ((mn, config) for mn, config in self.model_configs.items()
             if config.served_name == served_name),
            None,
        )
        if not match:
            self.log.warning(
                f"Slot {slot.slot_id}: pid {pid} serves '{served_name}' which is "
                f"not in MODEL_CONFIGS — skipping"
            )
            return
        model_name, config = match
        self.log.info(
            f"Slot {slot.slot_id}: found vLLM pid={pid} serving {served_name}, "
            f"waiting for /health (up to {self.ADOPT_BOOT_WAIT}s)"
        )
        # Wait for /health — it doesn't require auth and means vLLM is serving.
        deadline = time.monotonic() + self.ADOPT_BOOT_WAIT
        healthy = False
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                self.log.warning(
                    f"Slot {slot.slot_id}: vLLM pid={pid} died during boot — skipping"
                )
                return
            try:
                async with session.get(
                    f"http://127.0.0.1:{slot.port}/health"
                ) as r:
                    if r.status == 200:
                        healthy = True
                        break
            except Exception:
                pass
            await asyncio.sleep(2)
        if not healthy:
            self.log.warning(
                f"Slot {slot.slot_id}: pid {pid} /health not ready "
                f"within {self.ADOPT_BOOT_WAIT}s — skipping"
            )
            return
        b = self._make_backend(model_name, slot, served_name=served_name)
        await b.start()                # init session + idle watchdog
        b._adopted_pid = pid
        b._ready       = True
        slot.backend   = b
        self.log.info(
            f"Adopted vLLM on slot {slot.slot_id} (GPU={slot.gpu_id} "
            f"port={slot.port} model={model_name} pid={pid})"
        )


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    log = logging.getLogger("model_manager")

    slots  = [GpuSlot(sid, gid, port) for sid, gid, port in GPU_SLOTS]
    router = DynamicRouter(slots, MODEL_CONFIGS)

    app = web.Application(client_max_size=100 * 1024 * 1024)
    app.router.add_route("*", "/{path_info:.*}", router.handle)

    # Adopt any existing vLLM on slot ports before serving.  This keeps vLLM
    # warm across model_manager restarts (the systemd unit is KillMode=process).
    await router.adopt_existing_backends()

    # Background queue sampler that drives sustained-load scale-out.
    router.start_saturation_monitor()

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, LISTEN_HOST, LISTEN_PORT)
    await site.start()
    log.info(
        f"Listening on :{LISTEN_PORT} — "
        f"models={list(MODEL_CONFIGS)} slots={[str(s) for s in slots]}"
    )

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stop_event.set)
    try:
        await stop_event.wait()
    finally:
        # Stop the listener FIRST so no new requests can arrive and trigger a
        # spawn during shutdown (which previously orphaned a vLLM that we then
        # could not adopt).  Then tear down idle watchdogs and sessions.  vLLM
        # children are intentionally left running; next start adopts them.
        #
        # All phases are bounded so total shutdown is < 10s — keeps systemd's
        # state machine from getting confused during heavy in-flight load
        # (the "mm zombie" pattern: slow shutdown overlaps a queued restart,
        # the new instance races against the old one, and systemd loses
        # MainPID tracking).  Half-completed in-flight requests get cut off —
        # callers retry, which is fine.
        log.info("Shutdown — stopping HTTP listener")
        try:
            await asyncio.wait_for(site.stop(), timeout=5.0)
        except asyncio.TimeoutError:
            log.warning("site.stop() exceeded 5s — forcing exit")
        log.info("Shutdown — leaving vLLM backends running (will adopt on next start)")
        for slot in router._all_slots():
            if slot.backend and slot.backend._idle_task:
                slot.backend._idle_task.cancel()
            if slot.backend:
                try:
                    await asyncio.wait_for(slot.backend._close_session(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass
        try:
            await asyncio.wait_for(runner.cleanup(), timeout=3.0)
        except asyncio.TimeoutError:
            log.warning("runner.cleanup() exceeded 3s — forcing exit")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
