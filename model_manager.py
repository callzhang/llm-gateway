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
import json
import logging
import os
import signal
import subprocess
import time
import uuid

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

# Model configs: model_name → (startup_script, served_model_name)
# Add one entry per model you want to serve.  The startup script receives
# VLLM_CUDA_DEVICE and VLLM_PORT from model_manager at spawn time.
MODEL_CONFIGS: dict[str, tuple[str, str]] = {
    "qwen3.6-35b-a3b": ("run_qwen36_35b.sh", "qwen3.6-35b-a3b"),
    "qwen3.6-27b":     ("run_qwen36_27b.sh",  "qwen3.6-27b"),
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

        self.process: asyncio.subprocess.Process | None = None
        self._ready           = False
        self._failed          = False   # permanently dead; don't retry on this object
        self.last_activity    = time.monotonic()
        self._active_requests = 0
        self._lock            = asyncio.Lock()
        self._idle_task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None

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

                # Detect unexpected vLLM crash
                if self.process is not None and self.process.returncode is not None:
                    self.log.warning(
                        f"vLLM exited unexpectedly (rc={self.process.returncode}) — freeing slot"
                    )
                    self._ready   = False
                    self._failed  = True
                    self.process  = None
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
        """Raise RuntimeError if unexpected processes are occupying this GPU's VRAM.

        vLLM's EngineCore and worker sub-processes can escape process-group kills
        and linger with large CUDA allocations.  Catching this before launching
        produces a clean error instead of an inscrutable OOM 60s into startup.
        """
        try:
            # List every compute process on this specific GPU
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
                return   # nvidia-smi unavailable — skip check
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
            if not lines:
                return   # no compute processes on this GPU — good
            # Something is there.  Collect PIDs/usage and warn loudly.
            procs = []
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                pid_s  = parts[0] if parts else "?"
                mem_s  = parts[1] if len(parts) > 1 else "? MiB"
                procs.append(f"PID {pid_s} ({mem_s})")
            msg = (
                f"GPU {self.gpu_id} is occupied by unexpected process(es) before spawn: "
                + ", ".join(procs)
                + ". Kill them manually or wait for them to exit, then retry."
            )
            self.log.error(msg)
            # Do NOT mark _failed=True here — the caller will handle it.
            # Use a transient error so the slot is freed and a future request can retry.
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
            self.process = await asyncio.create_subprocess_exec(
                "bash", self.script,
                stdout=log_fd, stderr=log_fd,
                env=spawn_env,
                start_new_session=True,
            )
        finally:
            log_fd.close()

        deadline = time.monotonic() + WAKE_TIMEOUT
        started  = time.monotonic()
        while time.monotonic() < deadline:
            if self.process.returncode is not None:
                rc = self.process.returncode
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
        Caller must hold self._lock."""
        self._ready = False
        if self.process is None:
            return
        pid = self.process.pid
        # Always attempt to kill the entire process group, even if the APIServer
        # has already exited — orphan children (e.g. vLLM EngineCore) may still
        # hold ports or GPU memory and need to be explicitly reaped.
        self.log.info(f"Sending SIGTERM to pgid {pid}")
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            # Process group already gone — nothing to kill
            self.process = None
            return
        if self.process.returncode is None:
            # Main process still alive; wait for it to exit
            try:
                await asyncio.wait_for(self.process.wait(), timeout=30)
            except asyncio.TimeoutError:
                self.log.warning("SIGTERM timeout — sending SIGKILL")
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await self.process.wait()
        self.log.info("vLLM unloaded")
        self.process = None

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
        "max_tokens":  body.get("max_output_tokens", 4096),
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
        self._rr: dict[str, int] = {}          # round-robin counters per model
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
        """If all instances are saturated and a free slot exists, spawn another."""
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
                return

            slot   = free[0]
            script, served = self.model_configs[model_name]
            new_b  = GpuBackend(model_name, script, served, slot)
            slot.backend = new_b
            await new_b.start()

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
        if len(backends) == 1:
            return backends[0]
        model = backends[0].model_name
        idx   = self._rr.get(model, 0) % len(backends)
        self._rr[model] = idx + 1
        return backends[idx]

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

        prior_messages: list[dict] = []
        if prev_id := parsed.get("previous_response_id"):
            prior_messages = _response_store.get(prev_id, [])
            if not prior_messages:
                # Normal after a service restart — store is in-memory only
                self.log.debug(f"previous_response_id '{prev_id}' not found in store — starting fresh")

        completions_payload = _responses_to_completions(parsed, prior_messages)
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


# ── Entry point ────────────────────────────────────────────────────────────────

async def main() -> None:
    log = logging.getLogger("model_manager")

    slots  = [GpuSlot(sid, gid, port) for sid, gid, port in GPU_SLOTS]
    router = DynamicRouter(slots, MODEL_CONFIGS)

    app = web.Application(client_max_size=100 * 1024 * 1024)
    app.router.add_route("*", "/{path_info:.*}", router.handle)

    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", LISTEN_PORT).start()
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
        log.info("Shutdown — stopping all backends")
        for slot in slots:
            if slot.backend:
                await slot.backend.stop()
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
