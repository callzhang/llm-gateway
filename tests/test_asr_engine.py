import asyncio

import numpy as np
import pytest

from asr_adapter.audio import SAMPLE_RATE, Chunk
from asr_adapter.engine import EngineError, pcm16_wav, transcribe_chunks


def chunk(start, end):
    return Chunk(start, end, np.zeros(int((end - start) * SAMPLE_RATE), dtype=np.float32))


def run_transcribe(chunks, **kw):
    return asyncio.run(transcribe_chunks(chunks, **kw))


CHUNKS = [chunk(0, 30), chunk(30, 60), chunk(60, 90)]


class Recorder:
    def __init__(self, texts, fail_at=None, exc=EngineError("HTTP 503: gpu_busy")):
        self.texts, self.fail_at, self.exc, self.calls = list(texts), fail_at, exc, []

    async def __call__(self, ch):
        self.calls.append(ch.start)
        if self.fail_at is not None and len(self.calls) - 1 == self.fail_at:
            raise self.exc
        return self.texts[len(self.calls) - 1]


def test_gpu_serves_every_chunk_and_cpu_is_never_touched():
    gpu, cpu = Recorder(["一", "二", "三"]), Recorder([])
    result = run_transcribe(CHUNKS, gpu=gpu, cpu=cpu)
    assert [s["text"] for s in result.segments] == ["一", "二", "三"]
    assert cpu.calls == [] and result.engine == "gpu"


def test_a_busy_gpu_moves_the_whole_request_to_cpu():
    gpu, cpu = Recorder([], fail_at=0), Recorder(["甲", "乙", "丙"])
    result = run_transcribe(CHUNKS, gpu=gpu, cpu=cpu)
    assert [s["text"] for s in result.segments] == ["甲", "乙", "丙"]
    assert gpu.calls == [0] and cpu.calls == [0, 30, 60]      # GPU is not retried per chunk
    assert result.engine == "cpu"


def test_a_mid_request_gpu_failure_finishes_the_rest_on_cpu():
    gpu = Recorder(["一", "二"], fail_at=2)
    cpu = Recorder(["x", "y", "丙"])
    cpu.calls = [None, None]                                   # index the third CPU answer
    result = run_transcribe(CHUNKS, gpu=gpu, cpu=cpu)
    assert [s["text"] for s in result.segments] == ["一", "二", "丙"]
    assert result.engine == "gpu+cpu"


def test_without_a_gpu_engine_everything_runs_on_cpu():
    cpu = Recorder(["甲", "乙", "丙"])
    result = run_transcribe(CHUNKS, gpu=None, cpu=cpu)
    assert result.engine == "cpu" and len(result.segments) == 3


def test_timestamps_come_from_the_chunks_and_ids_are_sequential():
    result = run_transcribe(CHUNKS, gpu=None, cpu=Recorder(["甲", "", "丙"]))
    assert result.segments == [
        {"id": 0, "start": 0.0, "end": 30.0, "text": "甲"},
        {"id": 1, "start": 60.0, "end": 90.0, "text": "丙"},
    ]


def test_looping_output_is_dropped_and_reported_not_returned():
    loop = "我们是祖国的花朵，阳光下尽情唱着歌。" * 12
    result = run_transcribe(CHUNKS, gpu=None, cpu=Recorder(["好的，那我们开始吧。", loop, "谢谢大家"]))
    assert [s["text"] for s in result.segments] == ["好的，那我们开始吧。", "谢谢大家"]
    assert result.dropped == [{"start": 30.0, "end": 60.0}]


def test_non_engine_errors_are_not_swallowed():
    gpu = Recorder(["一"], fail_at=0, exc=ValueError("bug"))
    with pytest.raises(ValueError):
        run_transcribe(CHUNKS, gpu=gpu, cpu=Recorder([]))


def test_no_chunks_means_no_segments_and_no_engine():
    result = run_transcribe([], gpu=Recorder([]), cpu=Recorder([]))
    assert result.segments == [] and result.engine == "none"


def test_pcm16_wav_is_a_valid_16khz_mono_wav():
    import io, wave
    x = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    with wave.open(io.BytesIO(pcm16_wav(x))) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()) == (1, 2, SAMPLE_RATE, 5)
        frames = np.frombuffer(w.readframes(5), dtype="<i2")
    assert frames[0] == 0 and frames[1] > 0 > frames[2]
    assert frames[3] == 32767 and frames[4] == -32767          # clipped, not wrapped
