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
  IDLE_TIMEOUT   idle seconds before unload   (default: 300)
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
from collections import OrderedDict

import aiohttp
from aiohttp import web

# ── Configuration ──────────────────────────────────────────────────────────────
IDLE_TIMEOUT = int(os.environ.get("IDLE_TIMEOUT", "300"))
WAKE_TIMEOUT = int(os.environ.get("WAKE_TIMEOUT", "300"))
HEALTH_POLL  = float(os.environ.get("HEALTH_POLL", "2.0"))
LISTEN_PORT  = int(os.environ.get("LISTEN_PORT", "8002"))

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
    """Find a `vllm serve` process targeting this port, whether the port is
    bound yet or not.  Lets adoption recognise vLLM instances that are still
    cold-starting (model load takes ~30-60s; the API port is not bound until
    then).  Prefers the listening PID if present, else falls back to pgrep
    over the cmdline."""
    if pid := _find_pid_on_port(port):
        return pid
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"vllm serve.* --port {port}( |$)"],
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

# Model configs: model_name → (startup_script, served_model_name, allowed_gpu_ids)
#
# allowed_gpu_ids: set of GPU IDs this model is permitted to run on.
#   None  = no restriction (model works on any GPU in the pool).
#
# 35B-A3B used to be pinned to GPU 1 because the embedding-provider on GPU 0
# took ~2.5–3.5 GiB, leaving only ~28.5 GiB free vs the 29.16 GiB needed for
# gpu_memory_utilization=0.93 × 32 GiB.  Relaxing to None now: _check_gpu_free
# guards against actually-too-tight cases at spawn time, and most of the time
# GPU 0 has enough headroom to host a scale-out 35B replica.
#
# The startup script receives VLLM_CUDA_DEVICE and VLLM_PORT from model_manager
# at spawn time.
MODEL_CONFIGS: dict[str, tuple[str, str, "set[int] | None"]] = {
    "qwen3.6-35b-a3b":         ("run_qwen36_35b.sh",         "qwen3.6-35b-a3b",         None),
    "qwen3.6-35b-a3b-heretic": ("run_qwen36_35b_heretic.sh", "qwen3.6-35b-a3b-heretic", None),
    "qwen3.6-27b":             ("run_qwen36_27b.sh",         "qwen3.6-27b",             None),
}

# Minimum free GPU memory (GiB, from nvidia-smi) required to start a model.
# Used as a pre-eviction guard: before evicting an idle model to free a slot,
# check that the target GPU will have enough room after the eviction — this
# prevents destructively clearing a slot and then immediately failing the spawn.
# Rule of thumb: gpu_memory_utilization × GPU_total_GiB + 1 GiB safety buffer.
# If a model is not listed here no pre-check is performed (may evict & fail).
MODEL_MIN_FREE_GIB: dict[str, float] = {
    "qwen3.6-35b-a3b": 29.0,  # 0.93 × 32 GiB ≈ 29.8 GiB; lowered from 30.5
                               # GPU 0 shares with embedding-provider (~2.2 GiB),
                               # leaving only ~29.2 GiB free — actual vLLM usage
                               # is ~29.1 GiB so 29.0 threshold gives 0.2 GiB margin.
    "qwen3.6-35b-a3b-heretic": 29.0,  # same util=0.93 as stock 35b
    "qwen3.6-27b":     27.5,  # 0.84 × 32 GiB ≈ 26.9 + 0.6 GiB buffer
}

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

    def __init__(self, model_name: str, script: str, served_name: str, slot: GpuSlot):
        self.model_name  = model_name
        self.served_name = served_name
        self.slot        = slot
        self.vllm_port   = slot.port
        self.gpu_id      = slot.gpu_id
        self.vllm_base   = f"http://127.0.0.1:{slot.port}"
        self.script      = os.path.join(SCRIPT_DIR, script)
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
        self._session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=100, keepalive_timeout=60),
            timeout=aiohttp.ClientTimeout(total=None, connect=10, sock_read=None),
        )
        self._idle_task = asyncio.create_task(self._idle_loop())
        self.log.info(
            f"Backend claimed slot {self.slot.slot_id} "
            f"(GPU={self.gpu_id} port={self.vllm_port})"
        )

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
                idle = time.monotonic() - self.last_activity
                if idle < IDLE_TIMEOUT:
                    continue
                async with self._lock:
                    if not self.is_running or self._active_requests > 0:
                        continue
                    idle = time.monotonic() - self.last_activity
                    if idle >= IDLE_TIMEOUT:
                        self.log.info(f"Idle {int(idle)}s — unloading {self.model_name}")
                        await self._kill_process_locked()
                        if self.slot.backend is self:
                            self.slot.backend = None
                        await self._close_session()
                        return   # slot is free; exit watchdog
        except asyncio.CancelledError:
            pass

    # ── Process lifecycle ──────────────────────────────────────────────────────

    def _check_gpu_free(self) -> None:
        """Raise RuntimeError if a leftover vLLM process is occupying this GPU's VRAM.

        Only processes whose /proc/<pid>/cmdline contains 'vllm' are considered
        blockers.  Other legitimate GPU users (e.g. embedding servers) are ignored
        because they share VRAM without consuming the full allocation that vLLM needs.

        vLLM's EngineCore and worker sub-processes can escape process-group kills
        and linger with large CUDA allocations.  Catching this before launching
        produces a clean error instead of an inscrutable OOM 60s into startup.
        """
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader",
                    f"--id={self.gpu_id}",
                ],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if not lines:
                return

            vllm_procs = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                pid_s = parts[0] if parts else ""
                mem_s = parts[1] if len(parts) > 1 else "? MiB"
                if not pid_s.isdigit():
                    continue
                pid = int(pid_s)
                # Only flag processes that look like vLLM (cmdline contains 'vllm')
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        cmdline = f.read().replace(b"\x00", b" ").decode("utf-8", errors="ignore")
                    if "vllm" not in cmdline.lower():
                        continue   # unrelated GPU user — ignore
                except (FileNotFoundError, PermissionError):
                    continue   # process gone or not readable — skip
                vllm_procs.append(f"PID {pid} ({mem_s})")

            if not vllm_procs:
                return

            msg = (
                f"GPU {self.gpu_id} has leftover vLLM process(es) before spawn: "
                + ", ".join(vllm_procs)
                + ". Kill them manually or wait for them to exit, then retry."
            )
            self.log.error(msg)
            raise RuntimeError(msg)
        except FileNotFoundError:
            pass  # nvidia-smi not installed — skip check

    async def _spawn_locked(self) -> None:
        """Spawn vLLM subprocess and wait for /health.  Caller must hold self._lock."""
        self._ready = False
        self._check_gpu_free()
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
            }
            self.process = subprocess.Popen(
                ["bash", self.script],
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
                raise RuntimeError(
                    f"vLLM for '{self.model_name}' exited (rc={rc}) on slot {self.slot.slot_id}. "
                    f"See {self.log_path}."
                )
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
                        return
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
        # blocked by a zombie holding VRAM.
        await self._kill_gpu_zombies()
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

    async def _kill_gpu_zombies(self) -> None:
        """Kill any CUDA processes still holding GPU memory after a crash."""
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
                self.log.warning(f"Killing GPU {self.gpu_id} zombie PID {pid} (post-crash)")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
        except Exception as exc:
            self.log.warning(f"GPU zombie cleanup failed: {exc}")

    async def _forward(self, request: web.Request, body: bytes) -> web.StreamResponse:
        target_url  = f"{self.vllm_base}{request.path_qs}"
        fwd_headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in _HOP_BY_HOP | {"host", "content-length"}
        }
        # Rewrite routing key → vLLM's own --served-model-name
        if self.served_name != self.model_name and body:
            try:
                parsed = json.loads(body)
                if parsed.get("model") == self.model_name:
                    parsed["model"] = self.served_name
                    body = json.dumps(parsed).encode()
            except (json.JSONDecodeError, TypeError):
                pass
        try:
            async with self._session.request(
                method=request.method, url=target_url,
                headers=fwd_headers, data=body,
            ) as upstream:
                # ── EngineCore crash detection ──────────────────────────────
                # vLLM returns HTTP 500 with "EngineCore encountered an issue"
                # when its EngineCore subprocess crashes during inference.
                # The EngineCore process often survives the API server exit and
                # holds GPU memory indefinitely.  Detect this early, mark the
                # backend dead, and kill any lingering GPU processes.
                if upstream.status == 500:
                    err_body = await upstream.read()
                    if b"EngineCore" in err_body:
                        self.log.error(
                            f"vLLM EngineCore crash detected on slot {self.slot.slot_id} "
                            f"(GPU={self.gpu_id}) — marking backend dead"
                        )
                        self._ready = False
                        self._failed = True
                        if self.slot.backend is self:
                            self.slot.backend = None
                        asyncio.create_task(self._kill_gpu_zombies())
                    return web.Response(
                        status=500, content_type="application/json",
                        body=err_body,
                    )
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
            self.log.error(f"Cannot reach vLLM: {exc}")
            return web.Response(status=503, text="vLLM backend unavailable")
        except aiohttp.ClientError as exc:
            self.log.error(f"Proxy error: {exc}")
            return web.Response(status=502, text=f"Proxy error: {exc}")


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

    # Minimum number of simultaneous active requests across all running instances
    # before a scale-out is considered.  Setting this to 2 means a single
    # background heartbeat or probe never triggers a second GPU spawn; only real
    # concurrent user load does.
    SCALE_OUT_MIN_CONCURRENCY = 2

    # Seconds to wait before re-attempting scale-out after a failure.
    # Prevents a crash-loop when a slot cannot physically start a model
    # (e.g. not enough free VRAM because another process is on that GPU).
    SCALE_OUT_COOLDOWN = 120  # 2 minutes — short enough to recover after transient OOM

    def __init__(self, slots: list[GpuSlot], model_configs: dict[str, tuple[str, str, "set[int] | None"]]):
        self.slots         = slots
        self.model_configs = model_configs
        self.log           = logging.getLogger("mgr.router")
        self._router_lock  = asyncio.Lock()   # serialises slot claims
        self._scale_tasks: set[asyncio.Task] = set()
        # Per-model timestamp of last scale-out failure (monotonic clock).
        # Used to enforce SCALE_OUT_COOLDOWN before retrying.
        self._scale_fail_time: dict[str, float] = {}

    # ── Slot / backend helpers ─────────────────────────────────────────────────

    def _running_backends(self, model_name: str) -> list[GpuBackend]:
        """Backends that are fully ready to serve (spawned AND healthy)."""
        return [
            s.backend for s in self.slots
            if s.backend
            and s.backend.model_name == model_name
            and s.backend._ready
            and s.backend.is_running
        ]

    def _claimed_backends(self, model_name: str) -> list[GpuBackend]:
        """Running or mid-spawn, excluding permanently-failed ones."""
        return [
            s.backend for s in self.slots
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

    def _allowed_gpus(self, model_name: str) -> "set[int] | None":
        """GPU IDs this model may occupy, or None if unconstrained.

        Derived from the 3rd element of each MODEL_CONFIGS entry.  A None
        return means the model can run on any GPU in the slot pool.
        """
        cfg = self.model_configs.get(model_name)
        return cfg[2] if cfg is not None else None

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
                slot   = compatible[0]
                script, served, _ = self.model_configs[model_name]
                b      = GpuBackend(model_name, script, served, slot)
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

    async def _maybe_scale_out(self, model_name: str) -> None:
        """If all instances are saturated and a slot is available, spawn another.

        "Available" means either a truly free slot (no backend) OR a slot whose
        backend is a *different* model that is currently idle (0 active requests).
        In the latter case we evict the idle model first so the saturated model
        can use the slot.  This is important in a 2-slot system where both slots
        are always occupied by different models — without this, scale-out would
        never trigger even when one GPU is at 100% and the other is fully idle.
        """
        # Respect cooldown after a previous failure (prevents crash-loop when a
        # slot cannot physically start the model, e.g. insufficient free VRAM).
        last_fail = self._scale_fail_time.get(model_name, 0)
        if time.monotonic() - last_fail < self.SCALE_OUT_COOLDOWN:
            return

        running = self._running_backends(model_name)
        if not running:
            return
        total_active = sum(b._active_requests for b in running)
        if total_active < self.SCALE_OUT_MIN_CONCURRENCY:
            return   # not enough concurrent load to warrant a second GPU
        if any(b._active_requests == 0 for b in running):
            return   # at least one idle instance — no need to scale out

        slot:    GpuSlot     | None = None
        new_b:   GpuBackend  | None = None
        evict_b: GpuBackend  | None = None   # idle foreign backend to evict

        async with self._router_lock:
            running = self._running_backends(model_name)
            if not running:
                return
            total_active = sum(b._active_requests for b in running)
            if total_active < self.SCALE_OUT_MIN_CONCURRENCY:
                return
            if any(b._active_requests == 0 for b in running):
                return

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
                    self.log.info(
                        f"Scale-out for {model_name}: all free slots lack sufficient "
                        f"GPU memory (need {min_free_gib:.1f} GiB)"
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
                    return

                victim_slot = evictable[0]
                evict_b = victim_slot.backend   # save ref before we overwrite
                # Atomically claim the slot — prevents any other request from
                # grabbing it while we're killing the incumbent vLLM.
                victim_slot.backend = None      # detach old backend
                free = [victim_slot]

            slot   = free[0]
            script, served, _ = self.model_configs[model_name]
            new_b  = GpuBackend(model_name, script, served, slot)
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
            # Clear cooldown on success so future scale-outs can proceed promptly.
            self._scale_fail_time.pop(model_name, None)
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

        body = await request.read()
        model_name = self._extract_model(body)
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

        # Probe / latency-check detection (e.g. LiteLLM latency-based-routing).
        # Rule: only test models that are already loaded.
        #   • Model loaded  → let the request through for real latency measurement.
        #   • Model cold    → return 503 immediately, no spawn triggered.
        #     LiteLLM treats the 503 as "high latency / unavailable" and avoids
        #     routing to this model until it comes up naturally via a real request.
        try:
            parsed_body = json.loads(body)
            is_probe = (
                parsed_body.get("max_tokens", 9999) <= 1
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

        # Sticky routing: x-task-id header pins all turns of one task to the same
        # vLLM slot for prefix cache reuse.  Falls back to least-connections.
        try:
            parsed_body_for_sticky = json.loads(body) if body else None
        except Exception:
            parsed_body_for_sticky = None
        sticky_slot, sticky_reason = _sticky_slot_for(parsed_body_for_sticky, request.headers)
        sticky_backend: GpuBackend | None = None
        if sticky_slot is not None:
            sticky_backend = next(
                (b for b in backends if b.slot.slot_id == sticky_slot), None
            )
        if sticky_backend is not None:
            backend = sticky_backend
        else:
            backend = self._pick(backends)
            self._trigger_scale_out(model_name)
        # Record affinity so subsequent turns of the same task pin here.
        task_id = _extract_task_id(parsed_body_for_sticky, request.headers)
        # Approximate prompt size (sum of message-content chars) so the pipeline
        # team can grep-by-task and see whether multi-turn history is growing.
        msgs = (parsed_body_for_sticky or {}).get("messages", []) or []
        approx_chars = sum(
            len(m.get("content", "")) if isinstance(m.get("content"), str)
            else sum(len(p.get("text", "") or "") for p in (m.get("content") or []) if isinstance(p, dict))
            for m in msgs if isinstance(m, dict)
        )
        if task_id:
            hit_status = "hit" if sticky_backend is not None else "fresh"
            self.log.info(
                f"Sticky chat/completions: task_id={task_id} {hit_status} → slot {backend.slot.slot_id} "
                f"(msgs={len(msgs)}, chars≈{approx_chars})"
            )
            _set_task_affinity(task_id, backend.slot.slot_id)
        else:
            hdr_summary = ", ".join(
                f"{k.lower()}" for k in request.headers
                if k.lower().startswith(("x-", "anthropic-", "authorization", "user-agent"))
            )
            body_keys = sorted(parsed_body_for_sticky.keys()) if parsed_body_for_sticky else []
            self.log.info(
                f"chat/completions no task_id (headers: [{hdr_summary}], body keys: {body_keys}, "
                f"msgs={len(msgs)}, chars≈{approx_chars})"
            )
        return await backend.proxy(request, body)

    @staticmethod
    def _extract_model(body: bytes) -> str | None:
        if not body:
            return None
        try:
            return json.loads(body).get("model")
        except (json.JSONDecodeError, AttributeError):
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
            ((mn, script, served) for mn, (script, served, _allowed)
             in self.model_configs.items() if served == served_name),
            None,
        )
        if not match:
            self.log.warning(
                f"Slot {slot.slot_id}: pid {pid} serves '{served_name}' which is "
                f"not in MODEL_CONFIGS — skipping"
            )
            return
        model_name, script, _ = match
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
        b = GpuBackend(model_name, script, served_name, slot)
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

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", LISTEN_PORT)
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
        for slot in slots:
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
