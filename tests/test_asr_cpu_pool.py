"""CpuAsrWorker against a stand-in worker that speaks the real stdin/stdout protocol."""

import asyncio
import sys
import textwrap

import numpy as np
import pytest

from asr_adapter.audio import SAMPLE_RATE, Chunk
from asr_adapter.cpu_pool import CpuAsrWorker, CpuWorkerError

FAKE = textwrap.dedent('''
    import json, os, sys
    mode = os.environ.get("FAKE_MODE", "ok")
    out = sys.stdout
    if mode == "no_ready":
        sys.exit(1)
    out.write(json.dumps({"ready": True}) + "\\n"); out.flush()
    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break
        req = json.loads(line)
        stdin.read(req["samples"] * 4)
        if mode == "die":
            sys.exit(1)
        if mode == "error":
            out.write(json.dumps({"error": "boom"}) + "\\n")
        else:
            out.write(json.dumps({"text": f"n={req['samples']} lang={req['language']} max={req['max_new_tokens']}"}) + "\\n")
        out.flush()
''')


def chunk(seconds=2.0):
    return Chunk(0.0, seconds, np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32))


@pytest.fixture
def script(tmp_path):
    p = tmp_path / "fake_worker.py"
    p.write_text(FAKE)
    return p


def make(script, mode="ok", **kw):
    return CpuAsrWorker([sys.executable, str(script)], env={"FAKE_MODE": mode}, **kw)


def test_transcribes_over_the_protocol_and_reuses_the_process(script):
    async def go():
        w = make(script)
        try:
            a = await w.transcribe(chunk(2.0), "zh")
            pid = w._proc.pid
            b = await w.transcribe(chunk(1.0), None)
            assert w._proc.pid == pid                      # one warm process
            return a, b
        finally:
            await w.close()
    a, b = asyncio.run(go())
    assert a == "n=32000 lang=zh max=80" and b == "n=16000 lang=None max=72"


def test_a_worker_that_never_reports_ready_is_an_error(script):
    async def go():
        w = make(script, "no_ready", start_timeout=5)
        try:
            await w.transcribe(chunk(), None)
        finally:
            await w.close()
    with pytest.raises(CpuWorkerError, match="did not start"):
        asyncio.run(go())


def test_an_error_reply_fails_the_request_but_keeps_the_worker(script):
    async def go():
        w = make(script, "error")
        try:
            with pytest.raises(CpuWorkerError, match="boom"):
                await w.transcribe(chunk(), None)
            return w.running
        finally:
            await w.close()
    assert asyncio.run(go()) is True


def test_a_crash_mid_request_fails_and_the_next_request_starts_a_fresh_worker(script):
    async def go():
        w = make(script, "die")
        try:
            with pytest.raises(CpuWorkerError):
                await w.transcribe(chunk(), None)
            assert not w.running
            w._env = {"FAKE_MODE": "ok"}
            return await w.transcribe(chunk(1.0), None)
        finally:
            await w.close()
    assert asyncio.run(go()).startswith("n=16000")


def test_the_idle_worker_is_stopped(script):
    async def go():
        w = make(script, idle_seconds=1)
        try:
            await w.transcribe(chunk(), None)
            assert w.running
            await asyncio.sleep(2.6)
            return w.running
        finally:
            await w.close()
    assert asyncio.run(go()) is False


def test_requests_are_serialised(script):
    async def go():
        w = make(script)
        try:
            return await asyncio.gather(*(w.transcribe(chunk(1.0 + i), None) for i in range(3)))
        finally:
            await w.close()
    assert sorted(asyncio.run(go())) == ["n=16000 lang=None max=72", "n=32000 lang=None max=80",
                                          "n=48000 lang=None max=88"]
