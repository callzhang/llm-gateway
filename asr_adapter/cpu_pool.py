"""On-demand CPU worker owned by model_manager: spawned by the first request that needs
it, one inference at a time, terminated after `idle_seconds` without use."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Sequence

import numpy as np

from .audio import Chunk, token_budget

log = logging.getLogger("asr_adapter")


class CpuWorkerError(RuntimeError):
    """The CPU worker failed (crash, timeout, error reply).  There is no engine left to
    fall back to, so the request fails."""


class CpuAsrWorker:
    def __init__(self, cmd: Sequence[str], *, env: dict[str, str] | None = None,
                 idle_seconds: float = 900, start_timeout: float = 300, chunk_timeout: float = 180) -> None:
        self._cmd = list(cmd)
        self._env = env
        self._idle_seconds = idle_seconds
        self._start_timeout = start_timeout
        self._chunk_timeout = chunk_timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()                 # CPU-bound: one inference at a time
        self._last_used = time.monotonic()
        self._reaper: asyncio.Task | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def transcribe(self, chunk: Chunk, language: str | None) -> str:
        async with self._lock:
            await self._ensure()
            proc = self._proc
            assert proc is not None and proc.stdin is not None and proc.stdout is not None
            samples = np.ascontiguousarray(chunk.samples, dtype="<f4")
            header = {"samples": int(samples.size), "language": language,
                      "max_new_tokens": token_budget(chunk.end - chunk.start)}
            try:
                proc.stdin.write(json.dumps(header).encode() + b"\n" + samples.tobytes())
                await proc.stdin.drain()
                line = await asyncio.wait_for(proc.stdout.readline(), self._chunk_timeout)
            except (asyncio.TimeoutError, ConnectionError, BrokenPipeError) as exc:
                await self._kill()
                raise CpuWorkerError(f"CPU worker failed at {chunk.start:.1f}s: {type(exc).__name__}") from exc
            finally:
                self._last_used = time.monotonic()
            if not line:
                await self._kill()
                raise CpuWorkerError("CPU worker exited mid-request")
            reply = json.loads(line)
            if "error" in reply:
                raise CpuWorkerError(reply["error"])
            return str(reply.get("text") or "")

    async def _ensure(self) -> None:
        if self.running:
            return
        log.info("starting the CPU ASR worker")
        env = {**os.environ, **(self._env or {})}
        self._proc = await asyncio.create_subprocess_exec(
            *self._cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, env=env,
            limit=1 << 20,
        )
        try:
            line = await asyncio.wait_for(self._proc.stdout.readline(), self._start_timeout)
            ready = json.loads(line) if line else {}
        except (asyncio.TimeoutError, ValueError):
            ready = {}
        if not ready.get("ready"):
            await self._kill()
            raise CpuWorkerError("CPU worker did not start")
        self._last_used = time.monotonic()
        if self._reaper is None or self._reaper.done():
            self._reaper = asyncio.create_task(self._reap_idle())

    async def _reap_idle(self) -> None:
        while self.running:
            await asyncio.sleep(min(30.0, max(1.0, self._idle_seconds / 4)))
            if self._lock.locked():
                continue
            if time.monotonic() - self._last_used >= self._idle_seconds:
                async with self._lock:
                    if time.monotonic() - self._last_used >= self._idle_seconds:
                        log.info("stopping the idle CPU ASR worker")
                        await self._kill()
                        return

    async def _kill(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except Exception:
                pass
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

    async def close(self) -> None:
        if self._reaper is not None:
            self._reaper.cancel()
        async with self._lock:
            await self._kill()
