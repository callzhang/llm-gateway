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
import signal
import subprocess
import time
import uuid
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

# Model configs: model_name → (startup_script, served_model_name)
# Add one entry per model you want to serve.  The startup script receives
# VLLM_CUDA_DEVICE and VLLM_PORT from model_manager at spawn time.
MODEL_CONFIGS: dict[str, tuple[str, str]] = {
    "qwen3.6-35b-a3b": ("run_qwen36_35b.sh", "qwen3.6-35b-a3b"),
    "qwen3.6-27b":     ("run_qwen36_27b.sh",  "qwen3.6-27b"),
}

# Minimum free GPU memory (GiB, from nvidia-smi) required to start a model.
# Used as a pre-eviction guard: before evicting an idle model to free a slot,
# check that the target GPU will have enough room after the eviction — this
# prevents destructively clearing a slot and then immediately failing the spawn.
# Rule of thumb: gpu_memory_utilization × GPU_total_GiB + 1 GiB safety buffer.
# If a model is not listed here no pre-check is performed (may evict & fail).
MODEL_MIN_FREE_GIB: dict[str, float] = {
    "qwen3.6-35b-a3b": 30.5,  # 0.93 × 32 GiB ≈ 29.8 + 0.7 GiB buffer
    "qwen3.6-27b":     27.5,  # 0.84 × 32 GiB ≈ 26.9 + 0.6 GiB buffer
}

# Per-model context limits used for Responses API history trimming.
# (context_window_tokens, max_output_tokens)
# Keep in sync with vLLM --max-model-len and config.yaml max_tokens.
MODEL_LIMITS: dict[str, tuple[int, int]] = {
    "qwen3.6-35b-a3b": (122880, 32768),
    "qwen3.6-27b":     (65536,  16384),
}
# Fallback for models not listed above.
_DEFAULT_CONTEXT_WINDOW  = 32768
_DEFAULT_MAX_OUTPUT      = 4096

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
                    self._ready   = False
                    self._failed  = True
                    self.process  = None
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

        deadline = time.monotonic() + WAKE_TIMEOUT
        started  = time.monotonic()
        while time.monotonic() < deadline:
            # Guard against _idle_loop clearing self.process concurrently
            # (it runs without the lock; we're inside the lock but yield at await).
            if self.process is None or self._failed:
                raise RuntimeError(
                    f"vLLM for '{self.model_name}' crashed during startup "
                    f"(watchdog cleared process). See {self.log_path}."
                )
            rc = self.process.poll()
            if rc is not None:
                # Kill orphan children (e.g. EngineCore) that may still hold ports.
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
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
            except Exception:
                pass
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


# ── Responses API translation helpers ──────────────────────────────────────────

_response_store: dict[str, list[dict]] = {}

# ── Backend affinity for KV-cache reuse ───────────────────────────────────────
# Maps response_id → slot_id of the backend that served it.  Subsequent turns
# referencing this response_id via previous_response_id stick to the same slot
# so vLLM's per-instance prefix cache (--enable-prefix-caching) keeps hitting.
# OrderedDict gives FIFO eviction when the cap is reached.
_response_affinity: "OrderedDict[str, int]" = OrderedDict()
_AFFINITY_MAX = 10000


def _set_affinity(resp_id: str, slot_id: int) -> None:
    _response_affinity[resp_id] = slot_id
    _response_affinity.move_to_end(resp_id)
    while len(_response_affinity) > _AFFINITY_MAX:
        _response_affinity.popitem(last=False)

# ── Context-overflow circuit breaker ──────────────────────────────────────────
# Maps a stable request key → monotonic timestamp of last rejection.
# While the entry is younger than _CTX_CIRCUIT_TTL we short-circuit immediately
# (no payload build, no GPU allocation, no vLLM cold start).
_ctx_rejected: dict[str, float] = {}
_CTX_CIRCUIT_TTL = 300.0   # seconds — covers LiteLLM's default retry window


def _ctx_circuit_key(parsed: dict) -> str:
    """Return a stable key for the circuit breaker.

    Uses ``previous_response_id`` when present (exact conversation identity).
    Falls back to a hash of model + instructions + first 2 kB of input for
    fresh conversations that are already oversized.
    """
    prev_id = parsed.get("previous_response_id")
    if prev_id:
        return f"prev:{prev_id}"
    model = parsed.get("model", "")
    instr = parsed.get("instructions") or ""
    inp   = parsed.get("input", "")
    if not isinstance(inp, str):
        inp = json.dumps(inp, ensure_ascii=False)
    raw = f"{model}\x00{instr}\x00{inp[:2000]}"
    return f"hash:{hashlib.sha256(raw.encode()).hexdigest()[:16]}"


def _ctx_circuit_check(key: str) -> bool:
    """Return True if this key is tripped and the request should be rejected."""
    now = time.monotonic()
    # Lazy cleanup: sweep when the dict grows large
    if len(_ctx_rejected) > 200:
        expired = [k for k, ts in list(_ctx_rejected.items())
                   if now - ts > _CTX_CIRCUIT_TTL * 2]
        for k in expired:
            _ctx_rejected.pop(k, None)
    ts = _ctx_rejected.get(key)
    return ts is not None and (now - ts) < _CTX_CIRCUIT_TTL


def _ctx_circuit_trip(key: str) -> None:
    """Record a rejection so subsequent calls with the same key are fast-failed."""
    _ctx_rejected[key] = time.monotonic()

# Trim conversation history when estimated input token count exceeds this.
def _trim_to_fit(
    messages: list[dict],
    context_window: int,
    max_output_tokens: int,
    safety_margin: int = 512,
) -> tuple[list[dict], bool]:
    """Trim conversation history to fit within the model's context window.

    Conservative estimate: 1 char = 1 token (safe for Chinese/Japanese text).
    Trims oldest (user, assistant) pairs first to keep turns coherent.

    Returns:
        (trimmed, True)   — trimmed list fits within budget
        (original, False) — mandatory content (system + last user) alone exceeds
                            the budget; caller should return 400 immediately
    """
    def _char_count(m: dict) -> int:
        c = m.get("content", "")
        if isinstance(c, str):
            return len(c)
        if isinstance(c, list):
            return sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
        return 0

    # Cap at 70% of the context window to leave headroom for the model's own
    # output and avoid hitting vLLM's max_model_len edge.  Floor at half the
    # window so we never shrink absurdly when max_output_tokens is huge.
    budget = max(
        min(int(context_window * 0.7),
            context_window - max_output_tokens - safety_margin),
        context_window // 2,
    )

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system  = [m for m in messages if m.get("role") != "system"]

    if not non_system:
        return (messages, sum(_char_count(m) for m in messages) <= budget)

    last_msg = [non_system[-1]]   # current user turn — always kept
    history  = non_system[:-1]    # prior turns — eligible for trimming

    # System + current user must always fit; if not, fail fast
    mandatory_chars = sum(_char_count(m) for m in system_msgs + last_msg)
    if mandatory_chars > budget:
        return (messages, False)

    # Group history into (user, assistant) pairs; skip orphaned assistant at start
    pairs: list[tuple[dict, ...]] = []
    i = 0
    while i < len(history):
        role = history[i].get("role")
        if role == "user":
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                pairs.append((history[i], history[i + 1]))
                i += 2
            else:
                pairs.append((history[i],))
                i += 1
        else:
            # orphaned assistant or other role — skip
            i += 1

    # Include as many recent pairs as fit, working backwards from newest
    remaining = budget - mandatory_chars
    included: list[tuple[dict, ...]] = []
    for pair in reversed(pairs):
        pair_chars = sum(_char_count(m) for m in pair)
        if pair_chars > remaining:
            break   # this pair doesn't fit; drop it and everything older
        included.insert(0, pair)
        remaining -= pair_chars

    trimmed = system_msgs + [m for pair in included for m in pair] + last_msg
    return (trimmed, True)


def _responses_to_completions(body: dict, prior_messages: list[dict] | None = None) -> dict:
    messages: list[dict] = []
    instructions = body.get("instructions")

    if prior_messages:
        if instructions:
            messages.append({"role": "system", "content": instructions})
            messages.extend(m for m in prior_messages if m.get("role") != "system")
        else:
            messages.extend(prior_messages)
    elif instructions:
        messages.append({"role": "system", "content": instructions})

    input_val = body.get("input", [])
    if isinstance(input_val, str):
        messages.append({"role": "user", "content": input_val})
    elif isinstance(input_val, list):
        for item in input_val:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                continue
            role    = item.get("role", "user")
            content = item.get("content", "")
            if isinstance(content, list):
                text_parts:  list[str]  = []
                image_parts: list[dict] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype in ("input_text", "text", "output_text"):
                        text_parts.append(part.get("text", ""))
                    elif ptype in ("input_image", "image_url"):
                        url = part.get("image_url") or part.get("url") or ""
                        if isinstance(url, dict):
                            url = url.get("url", "")
                        if url:
                            image_parts.append({"type": "image_url", "image_url": {"url": url}})
                if image_parts:
                    content = image_parts + (
                        [{"type": "text", "text": "\n".join(text_parts)}] if text_parts else []
                    )
                else:
                    content = "\n".join(text_parts)
            messages.append({"role": role, "content": content})

    return {
        "model":       body.get("model", ""),
        "messages":    messages,
        "max_tokens":  body.get("max_output_tokens", 16384),
        "temperature": body.get("temperature", 1.0),
    }


def _completions_to_responses(chat_resp: dict | None, model_name: str, resp_id: str) -> dict:
    if not chat_resp:
        chat_resp = {}
    output_text = (chat_resp.get("choices") or [{}])[0].get("message", {}).get("content") or ""
    usage = chat_resp.get("usage", {})
    return {
        "id":         resp_id,
        "object":     "response",
        "created_at": int(time.time()),
        "model":      chat_resp.get("model", model_name),
        "status":     "completed",
        "output": [{
            "type":    "message",
            "id":      f"msg_{uuid.uuid4().hex}",
            "status":  "completed",
            "role":    "assistant",
            "content": [{"type": "output_text", "text": output_text, "annotations": []}],
        }],
        "output_text": output_text,
        "usage": {
            "input_tokens":  usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "total_tokens":  usage.get("total_tokens", 0),
        },
        "store": True,
    }


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

    def __init__(self, slots: list[GpuSlot], model_configs: dict[str, tuple[str, str]]):
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
                if not free:
                    occupied = [(s.slot_id, s.current_model) for s in self.slots
                                if not s.is_free]
                    raise GPUBusyError(
                        f"All GPU slots are occupied: {occupied}. "
                        f"Retry after {IDLE_TIMEOUT}s idle."
                    )
                slot   = free[0]
                script, served = self.model_configs[model_name]
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
            free = [s for s in self._free_slots() if s.slot_id not in running_slot_ids]

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
            script, served = self.model_configs[model_name]
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

        if request.method == "POST" and request.path in ("/v1/responses", "/responses"):
            return await self._handle_responses_api(request)

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

        # Probe detection — fake a success without cold-starting vLLM
        try:
            parsed_body = json.loads(body)
            is_probe = (
                parsed_body.get("max_tokens", 9999) <= 1
                and not parsed_body.get("messages", [{}])[-1].get("content", "").strip()
            )
            if is_probe:
                fake = {
                    "id": "probe-ok", "object": "chat.completion",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": ""},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                    "model": model_name,
                }
                return web.json_response(fake)
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

        backend = self._pick(backends)
        self._trigger_scale_out(model_name)
        return await backend.proxy(request, body)

    # ── Responses API ──────────────────────────────────────────────────────────

    async def _handle_responses_api(self, request: web.Request) -> web.StreamResponse:
        try:
            parsed: dict = json.loads(await request.read())
        except Exception:
            return web.Response(
                status=400, content_type="application/json",
                body=json.dumps({"error": {"message": "invalid JSON",
                                            "type": "invalid_request_error"}}),
            )

        model_name = parsed.get("model", "")
        if model_name not in self.model_configs:
            self.log.warning(f"Responses API: unknown model '{model_name}'")
            return web.Response(
                status=404, content_type="application/json",
                body=json.dumps({"error": {"message": f"Unknown model: {model_name}",
                                            "type": "invalid_request_error"}}),
            )

        # ── Circuit breaker: reject known-oversized requests immediately ──────
        # Checked before any store lookup or payload work so the same bad
        # request cannot repeatedly trigger a vLLM cold start.
        cb_key = _ctx_circuit_key(parsed)
        if _ctx_circuit_check(cb_key):
            err_msg = (
                f"Context window exceeded for model '{model_name}' — "
                f"this conversation is too long to continue. "
                f"Start a new conversation or reduce the message length."
            )
            self.log.warning(
                f"Responses API: circuit breaker tripped, rejecting without GPU work "
                f"(key={cb_key!r})"
            )
            return web.Response(
                status=400, content_type="application/json",
                headers={"x-should-retry": "false"},
                body=json.dumps({"error": {"message": err_msg,
                                            "type": "context_window_exceeded",
                                            "code": "context_length_exceeded"}}),
            )

        # ── Build payload and trim BEFORE spawning vLLM ───────────────────────
        # Fail fast with 400 if mandatory content already exceeds the context
        # window.  This prevents the infinite cold-start retry loop where an
        # oversized history causes vLLM to return 400, LiteLLM retries, the
        # retry triggers a fresh vLLM cold start, and so on.
        prior_messages: list[dict] = []
        if prev_id := parsed.get("previous_response_id"):
            prior_messages = _response_store.get(prev_id, [])
            if not prior_messages:
                # Normal after a service restart — store is in-memory only
                self.log.debug(
                    f"previous_response_id '{prev_id}' not found in store — starting fresh"
                )

        completions_payload = _responses_to_completions(parsed, prior_messages)

        ctx_win, max_out = MODEL_LIMITS.get(
            model_name, (_DEFAULT_CONTEXT_WINDOW, _DEFAULT_MAX_OUTPUT)
        )
        trimmed_msgs, fits = _trim_to_fit(
            completions_payload["messages"], ctx_win, max_out
        )
        if not fits:
            _ctx_circuit_trip(cb_key)
            err_msg = (
                f"The mandatory content (system prompt + current user message) "
                f"exceeds the context window for model '{model_name}' "
                f"({ctx_win} tokens). Please shorten your message."
            )
            self.log.warning(f"Responses API: {err_msg} (circuit breaker armed for {cb_key!r})")
            return web.Response(
                status=400, content_type="application/json",
                headers={"x-should-retry": "false"},
                body=json.dumps({"error": {"message": err_msg,
                                            "type": "context_window_exceeded",
                                            "code": "context_length_exceeded"}}),
            )
        completions_payload["messages"] = trimmed_msgs

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

        # Sticky routing: stick this conversation to the slot that served its
        # prior turn so vLLM's per-instance prefix cache keeps hitting.  Falls
        # back to least-connections if the slot is gone or now hosts a different
        # model (in which case it won't appear in the candidate list).
        sticky_backend: GpuBackend | None = None
        if prev_id := parsed.get("previous_response_id"):
            if (sticky_slot := _response_affinity.get(prev_id)) is not None:
                sticky_backend = next(
                    (b for b in backends if b.slot.slot_id == sticky_slot), None
                )
        if sticky_backend is not None:
            backend = sticky_backend
        else:
            backend = self._pick(backends)
            # Only trigger scale-out on cache miss: a sticky conversation
            # already has a home, scaling out for it wouldn't help.
            self._trigger_scale_out(model_name)
        # DEBUG-level so it's silent by default but available for diagnosing
        # KV-cache / sticky-routing effectiveness when needed.
        self.log.debug(
            f"Responses: model={model_name} "
            f"prev_id={'yes' if parsed.get('previous_response_id') else 'no'} "
            f"prior_msgs={len(prior_messages)} "
            f"msgs_sent={len(completions_payload['messages'])} → slot {backend.slot.slot_id}"
        )
        resp_id  = f"resp_{uuid.uuid4().hex}"
        stream   = bool(parsed.get("stream", False))
        target_url   = f"{backend.vllm_base}/v1/chat/completions"
        vllm_headers = {}
        if auth := request.headers.get("Authorization"):
            vllm_headers["Authorization"] = auth

        backend._active_requests += 1
        backend.last_activity = time.monotonic()
        try:
            if not stream:
                async with backend._session.post(target_url, json=completions_payload,
                                                  headers=vllm_headers) as upstream:
                    if upstream.status != 200:
                        data = await upstream.read()
                        return web.Response(status=upstream.status,
                                            content_type="application/json", body=data)
                    chat_resp = await upstream.json()
                result = _completions_to_responses(chat_resp, model_name, resp_id)
                output_text = result["output_text"]
                _response_store[resp_id] = completions_payload["messages"] + [
                    {"role": "assistant", "content": output_text}
                ]
                _set_affinity(resp_id, backend.slot.slot_id)
                return web.json_response(result)

            # ── Streaming ──────────────────────────────────────────────────────
            msg_id  = f"msg_{uuid.uuid4().hex}"
            created = int(time.time())

            def sse(event_type: str, payload: dict) -> bytes:
                return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n".encode()

            resp = web.StreamResponse(headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            })
            await resp.prepare(request)
            await resp.write(sse("response.created", {
                "type": "response.created",
                "response": {"id": resp_id, "object": "response", "created_at": created,
                             "model": model_name, "status": "in_progress",
                             "output": [], "output_text": ""},
            }))
            await resp.write(sse("response.output_item.added", {
                "type": "response.output_item.added", "output_index": 0,
                "item": {"id": msg_id, "type": "message", "status": "in_progress",
                         "role": "assistant", "content": []},
            }))
            await resp.write(sse("response.content_part.added", {
                "type": "response.content_part.added", "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }))

            full_text = ""
            usage: dict = {}
            streaming_payload = dict(completions_payload, stream=True)

            async with backend._session.post(target_url, json=streaming_payload,
                                              headers=vllm_headers) as upstream:
                if upstream.status != 200:
                    err = await upstream.read()
                    await resp.write(sse("error", {"type": "error",
                                                   "message": err.decode(errors="ignore")}))
                    await resp.write_eof()
                    return resp

                async for line_bytes in upstream.content:
                    line = line_bytes.decode("utf-8", errors="ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    piece = (choices[0].get("delta") or {}).get("content")
                    if piece:
                        full_text += piece
                        await resp.write(sse("response.output_text.delta", {
                            "type": "response.output_text.delta",
                            "item_id": msg_id, "output_index": 0,
                            "content_index": 0, "delta": piece,
                        }))

            await resp.write(sse("response.output_text.done", {
                "type": "response.output_text.done", "item_id": msg_id,
                "output_index": 0, "content_index": 0, "text": full_text,
            }))
            await resp.write(sse("response.content_part.done", {
                "type": "response.content_part.done", "item_id": msg_id,
                "output_index": 0, "content_index": 0,
                "part": {"type": "output_text", "text": full_text, "annotations": []},
            }))
            await resp.write(sse("response.output_item.done", {
                "type": "response.output_item.done", "output_index": 0,
                "item": {"id": msg_id, "type": "message", "status": "completed",
                         "role": "assistant",
                         "content": [{"type": "output_text", "text": full_text,
                                      "annotations": []}]},
            }))
            await resp.write(sse("response.completed", {
                "type": "response.completed",
                "response": {
                    "id": resp_id, "object": "response", "created_at": created,
                    "model": model_name, "status": "completed",
                    "output": [{"id": msg_id, "type": "message", "status": "completed",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": full_text,
                                             "annotations": []}]}],
                    "output_text": full_text,
                    "usage": {
                        "input_tokens":  usage.get("prompt_tokens", 0),
                        "output_tokens": usage.get("completion_tokens", 0),
                        "total_tokens":  usage.get("total_tokens", 0),
                    },
                },
            }))
            _response_store[resp_id] = completions_payload["messages"] + [
                {"role": "assistant", "content": full_text}
            ]
            _set_affinity(resp_id, backend.slot.slot_id)
            await resp.write_eof()
            return resp

        finally:
            backend._active_requests -= 1
            backend.last_activity = time.monotonic()

    @staticmethod
    def _extract_model(body: bytes) -> str | None:
        if not body:
            return None
        try:
            return json.loads(body).get("model")
        except (json.JSONDecodeError, AttributeError):
            return None

    # ── Adoption ───────────────────────────────────────────────────────────────

    ADOPT_BOOT_WAIT = 120   # max seconds to wait for a booting vLLM to expose /v1/models

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
            ((mn, script, served) for mn, (script, served)
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
        log.info("Shutdown — stopping HTTP listener")
        await site.stop()
        log.info("Shutdown — leaving vLLM backends running (will adopt on next start)")
        for slot in slots:
            if slot.backend and slot.backend._idle_task:
                slot.backend._idle_task.cancel()
            if slot.backend:
                await slot.backend._close_session()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
