# vLLM as Independent systemd Services

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple vLLM process lifetime from model_manager so that restarting model_manager does not kill running inference, and so that vLLM is stopped gracefully (SIGTERM → drain → SIGKILL) rather than by killing a subprocess.

**Architecture:** Each (model × GPU-slot) combination becomes a pre-defined user-level systemd service. model_manager controls them via `systemctl --user start/stop` instead of `subprocess.Popen`. On startup, model_manager scans for already-running services and adopts them (no cold start after a model_manager-only restart). A background watchdog in every `GpuBackend` pings `/health` + queries `systemctl is-active` every 30 s; on failure it cleans GPU zombie processes and frees the slot.

**Tech Stack:** Python asyncio, aiohttp, systemd user services, nvidia-smi

---

## File map

| File | Change |
|------|--------|
| `~/.config/systemd/user/llm-vllm-35b-slot0.service` | CREATE — qwen3.6-35b-a3b on GPU 0 port 9000 |
| `~/.config/systemd/user/llm-vllm-35b-slot1.service` | CREATE — qwen3.6-35b-a3b on GPU 1 port 9010 |
| `~/.config/systemd/user/llm-vllm-27b-slot0.service` | CREATE — qwen3.6-27b on GPU 0 port 9000 |
| `~/.config/systemd/user/llm-vllm-27b-slot1.service` | CREATE — qwen3.6-27b on GPU 1 port 9010 |
| `model_manager.py` | MODIFY — refactor GpuBackend + DynamicRouter |

`MODEL_CONFIGS` gains a third element per entry — the systemd service name prefix
(e.g. `"llm-vllm-35b"`). Full service name = `f"{prefix}-slot{slot_id}"`.

---

## Task 1: Create vLLM systemd service files

**Files:**
- Create: `~/.config/systemd/user/llm-vllm-35b-slot0.service`
- Create: `~/.config/systemd/user/llm-vllm-35b-slot1.service`
- Create: `~/.config/systemd/user/llm-vllm-27b-slot0.service`
- Create: `~/.config/systemd/user/llm-vllm-27b-slot1.service`

- [ ] **Step 1: Write the four service files**

`~/.config/systemd/user/llm-vllm-35b-slot0.service`:
```ini
[Unit]
Description=vLLM — qwen3.6-35b-a3b slot 0 (GPU 0 :9000)
# Deliberately no After= / Requires= — model_manager starts/stops this on demand

[Service]
Type=simple
WorkingDirectory=/home/derek/Projects/llm-gateway
Environment=VLLM_CUDA_DEVICE=0
Environment=VLLM_PORT=9000
ExecStart=/home/derek/Projects/llm-gateway/run_qwen36_35b.sh
# Kill the entire cgroup so EngineCore sub-process also dies
KillMode=control-group
KillSignal=SIGTERM
# Give vLLM 60 s to drain in-flight requests before SIGKILL
TimeoutStopSec=60
# Never auto-restart — model_manager owns the lifecycle
Restart=no
StandardOutput=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-35b-a3b_slot0.log
StandardError=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-35b-a3b_slot0.log

[Install]
# Not enabled at boot — started on demand by model_manager
```

`~/.config/systemd/user/llm-vllm-35b-slot1.service`:
```ini
[Unit]
Description=vLLM — qwen3.6-35b-a3b slot 1 (GPU 1 :9010)

[Service]
Type=simple
WorkingDirectory=/home/derek/Projects/llm-gateway
Environment=VLLM_CUDA_DEVICE=1
Environment=VLLM_PORT=9010
ExecStart=/home/derek/Projects/llm-gateway/run_qwen36_35b.sh
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=60
Restart=no
StandardOutput=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-35b-a3b_slot1.log
StandardError=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-35b-a3b_slot1.log
```

`~/.config/systemd/user/llm-vllm-27b-slot0.service`:
```ini
[Unit]
Description=vLLM — qwen3.6-27b slot 0 (GPU 0 :9000)

[Service]
Type=simple
WorkingDirectory=/home/derek/Projects/llm-gateway
Environment=VLLM_CUDA_DEVICE=0
Environment=VLLM_PORT=9000
ExecStart=/home/derek/Projects/llm-gateway/run_qwen36_27b.sh
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=60
Restart=no
StandardOutput=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-27b_slot0.log
StandardError=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-27b_slot0.log
```

`~/.config/systemd/user/llm-vllm-27b-slot1.service`:
```ini
[Unit]
Description=vLLM — qwen3.6-27b slot 1 (GPU 1 :9010)

[Service]
Type=simple
WorkingDirectory=/home/derek/Projects/llm-gateway
Environment=VLLM_CUDA_DEVICE=1
Environment=VLLM_PORT=9010
ExecStart=/home/derek/Projects/llm-gateway/run_qwen36_27b.sh
KillMode=control-group
KillSignal=SIGTERM
TimeoutStopSec=60
Restart=no
StandardOutput=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-27b_slot1.log
StandardError=append:/home/derek/Projects/llm-gateway/logs/qwen3_6-27b_slot1.log
```

- [ ] **Step 2: Reload systemd and verify units are known**

```bash
systemctl --user daemon-reload
systemctl --user list-unit-files 'llm-vllm-*.service'
```
Expected output (4 lines, all `static` or `disabled`):
```
llm-vllm-27b-slot0.service  static  -
llm-vllm-27b-slot1.service  static  -
llm-vllm-35b-slot0.service  static  -
llm-vllm-35b-slot1.service  static  -
```

- [ ] **Step 3: Verify manual start/stop works for one service**

```bash
# Stop model_manager first so its subprocess doesn't conflict
systemctl --user stop llm-model-manager.service

# Start one vLLM service and watch it come up
systemctl --user start llm-vllm-35b-slot0.service
systemctl --user status llm-vllm-35b-slot0.service
# Wait for /health (takes ~90s cold start)
for i in $(seq 60); do curl -sf http://127.0.0.1:9000/health && echo "ready" && break; sleep 3; done

# Stop it gracefully
systemctl --user stop llm-vllm-35b-slot0.service
# Check exit code in journal
journalctl --user -u llm-vllm-35b-slot0.service --no-pager -n 5
```
Expected: service active during test, stopped cleanly (no SIGKILL in journal).

- [ ] **Step 4: Restart model_manager**

```bash
systemctl --user start llm-model-manager.service
```

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "Add independent systemd service files for vLLM slots"
```

---

## Task 2: Extend MODEL_CONFIGS and add service name to GpuBackend

**Files:**
- Modify: `model_manager.py` — `MODEL_CONFIGS`, `GpuBackend.__init__`

- [ ] **Step 1: Add service_prefix to MODEL_CONFIGS**

In `model_manager.py`, find:
```python
MODEL_CONFIGS: dict[str, tuple[str, str]] = {
    "qwen3.6-35b-a3b": ("run_qwen36_35b.sh", "qwen3.6-35b-a3b"),
    "qwen3.6-27b":     ("run_qwen36_27b.sh",  "qwen3.6-27b"),
}
```
Replace with:
```python
# (startup_script, served_model_name, systemd_service_prefix)
# Full service unit = f"{prefix}-slot{slot_id}.service"
MODEL_CONFIGS: dict[str, tuple[str, str, str]] = {
    "qwen3.6-35b-a3b": ("run_qwen36_35b.sh", "qwen3.6-35b-a3b", "llm-vllm-35b"),
    "qwen3.6-27b":     ("run_qwen36_27b.sh",  "qwen3.6-27b",    "llm-vllm-27b"),
}
```

- [ ] **Step 2: Update GpuBackend.__init__ to accept and store service_prefix**

Find:
```python
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
```
Replace with:
```python
    def __init__(self, model_name: str, script: str, served_name: str,
                 service_prefix: str, slot: GpuSlot):
        self.model_name    = model_name
        self.served_name   = served_name
        self.slot          = slot
        self.vllm_port     = slot.port
        self.gpu_id        = slot.gpu_id
        self.vllm_base     = f"http://127.0.0.1:{slot.port}"
        self.script        = os.path.join(SCRIPT_DIR, script)
        self.service_name  = f"{service_prefix}-slot{slot.slot_id}"
        safe               = model_name.replace(".", "_")
        self.log_path      = os.path.join(LOG_DIR, f"{safe}_slot{slot.slot_id}.log")
        self.log           = logging.getLogger(f"mgr.s{slot.slot_id}.{model_name}")
```

- [ ] **Step 3: Remove self.process from __init__**

Also in `__init__`, remove:
```python
        self.process: asyncio.subprocess.Process | None = None
```

- [ ] **Step 4: Update is_running property**

Find:
```python
    @property
    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None
```
Replace with:
```python
    @property
    def is_running(self) -> bool:
        """True when this backend has been confirmed ready via /health.
        The systemd service state is checked by the background watchdog."""
        return self._ready and not self._failed
```

- [ ] **Step 5: Update all three GpuBackend construction sites in DynamicRouter**

There are three places that construct a `GpuBackend`. All need the new `service_prefix` argument.

In `_get_or_start()`:
```python
# OLD
script, served = self.model_configs[model_name]
b      = GpuBackend(model_name, script, served, slot)

# NEW
script, served, svc_prefix = self.model_configs[model_name]
b = GpuBackend(model_name, script, served, svc_prefix, slot)
```

In `_maybe_scale_out()`:
```python
# OLD
script, served = self.model_configs[model_name]
new_b  = GpuBackend(model_name, script, served, slot)

# NEW
script, served, svc_prefix = self.model_configs[model_name]
new_b = GpuBackend(model_name, script, served, svc_prefix, slot)
```

- [ ] **Step 6: Verify syntax**

```bash
python3 -m py_compile model_manager.py && echo OK
```
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add model_manager.py
git commit -m "Add service_prefix to MODEL_CONFIGS and GpuBackend"
```

---

## Task 3: Replace subprocess spawn/kill with systemctl start/stop

**Files:**
- Modify: `model_manager.py` — replace `_spawn_locked` and `_kill_process_locked`

- [ ] **Step 1: Replace _spawn_locked with _start_service_locked**

Find and delete the entire `_spawn_locked` method (from `async def _spawn_locked` to the closing `raise RuntimeError`). Replace with:

```python
    async def _start_service_locked(self) -> None:
        """Start the vLLM systemd service and poll /health until ready."""
        self._ready = False
        self._check_gpu_free()

        self.log.info(
            f"Starting {self.service_name} (GPU={self.gpu_id} port={self.vllm_port})"
        )
        start_proc = await asyncio.create_subprocess_exec(
            "systemctl", "--user", "start", self.service_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await start_proc.communicate()
        if start_proc.returncode != 0:
            raise RuntimeError(
                f"systemctl start {self.service_name} failed: {stderr.decode().strip()}"
            )

        deadline = time.monotonic() + WAKE_TIMEOUT
        started  = time.monotonic()
        while time.monotonic() < deadline:
            # Bail early if the service entered failed state
            chk = await asyncio.create_subprocess_exec(
                "systemctl", "--user", "is-failed", "--quiet", self.service_name,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await chk.wait()
            if chk.returncode == 0:
                raise RuntimeError(
                    f"Service {self.service_name} entered failed state. "
                    f"Run: journalctl --user -u {self.service_name}"
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

        self.log.error(f"Startup timed out after {WAKE_TIMEOUT}s — stopping service")
        await self._stop_service_locked()
        raise RuntimeError(
            f"Service {self.service_name} did not become healthy within {WAKE_TIMEOUT}s."
        )
```

- [ ] **Step 2: Replace _kill_process_locked with _stop_service_locked**

Find and delete the entire `_kill_process_locked` method. Replace with:

```python
    async def _stop_service_locked(self) -> None:
        """Stop the vLLM systemd service gracefully (SIGTERM → drain → SIGKILL).
        systemd handles the timeout and escalation via TimeoutStopSec in the unit file."""
        self._ready = False
        self.log.info(f"Stopping {self.service_name} (graceful drain via systemctl)")
        stop_proc = await asyncio.create_subprocess_exec(
            "systemctl", "--user", "stop", self.service_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await stop_proc.communicate()
        # Reap any GPU zombie processes the service may have left behind
        await self._kill_gpu_zombies()
        self.log.info(f"{self.service_name} stopped")
```

- [ ] **Step 3: Update _ensure_running to call _start_service_locked**

Find:
```python
            try:
                await self._spawn_locked()
            except Exception:
                self._failed = True
                raise
```
Replace with:
```python
            try:
                await self._start_service_locked()
            except Exception:
                self._failed = True
                raise
```

- [ ] **Step 4: Update stop() to call _stop_service_locked**

Find:
```python
    async def stop(self) -> None:
        """Gracefully stop this backend and release its slot."""
        if self._idle_task:
            self._idle_task.cancel()
        async with self._lock:
            await self._kill_process_locked()
        await self._close_session()
        if self.slot.backend is self:
            self.slot.backend = None
```
Replace with:
```python
    async def stop(self) -> None:
        """Gracefully stop this backend and release its slot."""
        if self._idle_task:
            self._idle_task.cancel()
        async with self._lock:
            await self._stop_service_locked()
        await self._close_session()
        if self.slot.backend is self:
            self.slot.backend = None
```

- [ ] **Step 5: Update idle_loop unload path to call _stop_service_locked**

Find the idle timeout unload block in `_idle_loop`:
```python
                    if idle >= IDLE_TIMEOUT:
                        self.log.info(f"Idle {int(idle)}s — unloading {self.model_name}")
                        await self._kill_process_locked()
                        if self.slot.backend is self:
                            self.slot.backend = None
                        await self._close_session()
                        return   # slot is free; exit watchdog
```
Replace with:
```python
                    if idle >= IDLE_TIMEOUT:
                        self.log.info(f"Idle {int(idle)}s — unloading {self.model_name}")
                        await self._stop_service_locked()
                        if self.slot.backend is self:
                            self.slot.backend = None
                        await self._close_session()
                        return
```

- [ ] **Step 6: Remove now-dead imports (signal module may still be used by _kill_gpu_zombies)**

Check if `signal` and `subprocess` are still needed:
```bash
grep -n "^import signal\|os\.killpg\|os\.kill\|signal\.SIG" model_manager.py | head -20
```
Keep both — `_kill_gpu_zombies` still uses `os.killpg` / `signal.SIGKILL`.

- [ ] **Step 7: Verify syntax and restart**

```bash
python3 -m py_compile model_manager.py && echo OK
systemctl --user restart llm-model-manager.service
sleep 3 && systemctl --user is-active llm-model-manager.service
```
Expected: `OK`, then `active`

- [ ] **Step 8: Smoke test — make a request and verify vLLM starts via systemctl**

```bash
# In one terminal, watch service state
watch -n2 'systemctl --user is-active llm-vllm-35b-slot0.service'

# In another terminal, send a request
curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-local-gateway" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hi"}],"max_tokens":50}' \
  | python3 -m json.tool
```
Expected: `llm-vllm-35b-slot0.service` transitions `inactive → activating → active`, response contains `"content"` with text.

- [ ] **Step 9: Commit**

```bash
git add model_manager.py
git commit -m "Replace subprocess spawn/kill with systemctl start/stop"
```

---

## Task 4: Continuous health monitoring in _idle_loop

**Files:**
- Modify: `model_manager.py` — `_idle_loop`

The current watchdog only checks `process.returncode` every 30 s. With systemd, we replace that with HTTP `/health` + `systemctl is-active` checks. A failed check triggers GPU zombie cleanup and slot release.

- [ ] **Step 1: Rewrite the dead-process detection block in _idle_loop**

Find the crash detection block (currently checks `process.returncode`):
```python
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
```
Replace with:
```python
                # ── Continuous health check ─────────────────────────────────
                # Only run when we believe the service is up (_ready=True).
                if self._ready:
                    # 1. HTTP /health — definitive liveness check
                    http_ok = False
                    try:
                        async with self._session.get(
                            f"{self.vllm_base}/health",
                            timeout=aiohttp.ClientTimeout(total=5),
                        ) as r:
                            http_ok = (r.status == 200)
                    except Exception:
                        pass

                    # 2. systemd unit state — catches silent process deaths
                    svc_chk = await asyncio.create_subprocess_exec(
                        "systemctl", "--user", "is-active", "--quiet",
                        self.service_name,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                    await svc_chk.wait()
                    svc_active = (svc_chk.returncode == 0)

                    if not http_ok or not svc_active:
                        self.log.warning(
                            f"{self.service_name} unhealthy "
                            f"(http={http_ok} svc={svc_active}) — freeing slot"
                        )
                        self._ready  = False
                        self._failed = True
                        if self.slot.backend is self:
                            self.slot.backend = None
                        asyncio.create_task(self._kill_gpu_zombies())
                        await self._close_session()
                        return
```

- [ ] **Step 2: Verify syntax**

```bash
python3 -m py_compile model_manager.py && echo OK
```

- [ ] **Step 3: Test crash detection**

```bash
# Trigger a request so vLLM starts (takes ~90s cold start)
curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H "Authorization: Bearer sk-local-gateway" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}],"max_tokens":10}' \
  | python3 -m json.tool

# Forcefully kill vLLM outside of model_manager
systemctl --user kill --kill-who=all --signal=SIGKILL llm-vllm-35b-slot0.service

# Watch model_manager log for watchdog detection (within 30s)
tail -f logs/model_manager.log | grep -E "unhealthy|freeing slot|failed"
```
Expected: within 30 s, log shows `llm-vllm-35b-slot0 unhealthy ... freeing slot`.

- [ ] **Step 4: Commit**

```bash
git add model_manager.py
git commit -m "Continuous health monitoring: HTTP + systemctl in _idle_loop"
```

---

## Task 5: Adopt already-running services on startup

**Files:**
- Modify: `model_manager.py` — add `DynamicRouter.adopt_running_services()`, call from `main()`

This makes model_manager restart transparent: if vLLM is already running, no cold start.

- [ ] **Step 1: Add adopt_running_services to DynamicRouter**

Add after the `__init__` method of `DynamicRouter` (before `_running_backends`):

```python
    async def adopt_running_services(self) -> None:
        """Detect and adopt already-running vLLM services.

        Called once at startup.  If model_manager was restarted while vLLM was
        serving, this reconstructs the in-memory slot state so requests are
        routed to the still-running instances without a cold start.
        """
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=5)
        ) as probe:
            for slot in self.slots:
                if slot.backend is not None:
                    continue  # already claimed
                for model_name, (script, served, svc_prefix) in self.model_configs.items():
                    svc_name = f"{svc_prefix}-slot{slot.slot_id}"

                    # Check systemd unit state (synchronous — startup only)
                    chk = subprocess.run(
                        ["systemctl", "--user", "is-active", "--quiet", svc_name],
                        capture_output=True,
                    )
                    if chk.returncode != 0:
                        continue

                    # Verify HTTP health
                    try:
                        async with probe.get(
                            f"http://127.0.0.1:{slot.port}/health"
                        ) as r:
                            if r.status != 200:
                                continue
                    except Exception:
                        continue

                    b = GpuBackend(model_name, script, served, svc_prefix, slot)
                    slot.backend = b
                    await b.start()   # init session + watchdog
                    b._ready = True
                    self.log.info(
                        f"Adopted running service {svc_name} "
                        f"(slot {slot.slot_id} GPU={slot.gpu_id})"
                    )
                    break   # one model per slot
```

- [ ] **Step 2: Call adopt_running_services in main() before starting the HTTP server**

Find in `main()` (or the app startup section) where the router is created. It will look something like:

```python
    slots  = [GpuSlot(sid, gid, port) for sid, gid, port in GPU_SLOTS]
    router = DynamicRouter(slots, MODEL_CONFIGS)
    app    = web.Application()
    ...
    web.run_app(app, ...)
```

Add the adoption call after router creation:

```python
    slots  = [GpuSlot(sid, gid, port) for sid, gid, port in GPU_SLOTS]
    router = DynamicRouter(slots, MODEL_CONFIGS)
    # Adopt any vLLM services still running from before a model_manager restart
    loop = asyncio.get_event_loop()
    loop.run_until_complete(router.adopt_running_services())
    ...
```

If `main()` is already async (uses `asyncio.run`), use `await router.adopt_running_services()` directly.

- [ ] **Step 3: Verify syntax**

```bash
python3 -m py_compile model_manager.py && echo OK
```

- [ ] **Step 4: Test adoption end-to-end**

```bash
# 1. Start a model so vLLM is running
curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H "Authorization: Bearer sk-local-gateway" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}],"max_tokens":10}' \
  | python3 -m json.tool

# 2. Verify vLLM service is active
systemctl --user is-active llm-vllm-35b-slot0.service

# 3. Restart ONLY model_manager (not vLLM)
systemctl --user restart llm-model-manager.service
sleep 5

# 4. Check log for adoption message (no cold start)
grep "Adopted" logs/model_manager.log | tail -5

# 5. Send another request — should respond quickly (no 90s cold start)
time curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H "Authorization: Bearer sk-local-gateway" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}],"max_tokens":10}' \
  | python3 -m json.tool
```
Expected: log shows `Adopted running service llm-vllm-35b-slot0`, second request responds in < 10 s.

- [ ] **Step 5: Commit**

```bash
git add model_manager.py
git commit -m "Adopt already-running vLLM services on model_manager startup"
```

---

## Task 6: Remove KillMode=control-group from llm-model-manager.service

**Files:**
- Modify: `~/.config/systemd/user/llm-model-manager.service`

Currently model_manager's service has `KillMode=control-group`, which is what kills vLLM when model_manager stops. Now that vLLM runs as independent services, this should be `KillMode=mixed` (send SIGTERM to main process only) or `process`.

- [ ] **Step 1: Edit llm-model-manager.service**

In `~/.config/systemd/user/llm-model-manager.service`, change:
```ini
KillMode=control-group
```
to:
```ini
KillMode=process
```

- [ ] **Step 2: Reload and verify**

```bash
systemctl --user daemon-reload
systemctl --user restart llm-model-manager.service
systemctl --user is-active llm-model-manager.service
```

- [ ] **Step 3: Final end-to-end test**

```bash
# 1. Cold-start vLLM via a request
curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H "Authorization: Bearer sk-local-gateway" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}],"max_tokens":50}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"

# 2. Restart model_manager — vLLM must survive
systemctl --user restart llm-model-manager.service
sleep 5
systemctl --user is-active llm-vllm-35b-slot0.service   # must still be: active

# 3. Request again — must be fast (adopted, no cold start)
time curl -s -X POST http://127.0.0.1:8901/v1/chat/completions \
  -H "Authorization: Bearer sk-local-gateway" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"hello"}],"max_tokens":50}' \
  | python3 -c "import json,sys; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])"
```
Expected: first request ~90s (cold start), model_manager restart < 5s with vLLM still active, second request < 5s.

- [ ] **Step 4: Commit and push**

```bash
git add model_manager.py
git commit -m "Set KillMode=process on model_manager — vLLM survives restart"
git push
```

---

## Self-review

**Spec coverage:**
- ✅ vLLM as independent services (Task 1)
- ✅ model_manager controls via systemctl (Tasks 2–3)
- ✅ Graceful drain: `TimeoutStopSec=60` + `KillMode=control-group` in vLLM services
- ✅ Dead process detection via HTTP + systemctl (Task 4)
- ✅ GPU zombie cleanup on failure (existing `_kill_gpu_zombies`, wired in Task 4)
- ✅ Startup adoption — no cold start after model_manager restart (Task 5)
- ✅ model_manager restart no longer kills vLLM (Task 6)

**Gaps / notes:**
- `_kill_gpu_zombies` is `async` in the current code but calls `subprocess.run` synchronously.  
  That is fine for occasional zombie cleanup; no change needed.
- The `signal` import stays because `_kill_gpu_zombies` uses `os.killpg(pid, signal.SIGKILL)` for orphaned processes that survive even `systemctl stop`.
- Service files have no `[Install]` section and are not enabled — they are started on demand only. If you ever want vLLM to start at boot independently, add `WantedBy=default.target` and `systemctl --user enable`.
