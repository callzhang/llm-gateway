"""Engine selection for one transcription request: GPU first, CPU as the fallback.

The GPU engine is Qwen3-ASR-0.6B FP8 scheduled by model_manager as a co-resident lane
(fast, but only when a GPU has ~3.7 GiB free); the CPU engine is the same model in a
child process (always available, ~6x realtime).  Both engines are async callables
chunk -> text, so the policy is testable without a model.  The policy is per request:
once the GPU declines (busy, cold-start failure, error) the rest of the request stays
on CPU instead of retrying the GPU chunk by chunk.
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import numpy as np

from .audio import SAMPLE_RATE, Chunk, is_degenerate

log = logging.getLogger("asr_adapter")

Engine = Callable[[Chunk], Awaitable[str]]


class EngineError(RuntimeError):
    """An engine could not transcribe (busy, unreachable, HTTP error).  Recoverable by
    falling back; anything else (a bug) must propagate."""


@dataclass
class TranscriptionResult:
    segments: list[dict] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)
    engine: str = "none"          # "gpu", "cpu", "gpu+cpu" or "none"


def pcm16_wav(samples: np.ndarray) -> bytes:
    """16 kHz mono PCM16 WAV bytes (samples clipped to [-1, 1])."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


async def transcribe_chunks(
    chunks: list[Chunk], *, gpu: Engine | None, cpu: Engine,
) -> TranscriptionResult:
    result = TranscriptionResult()
    used: set[str] = set()
    use_gpu = gpu is not None
    for chunk in chunks:
        text: str | None = None
        if use_gpu:
            try:
                text = await gpu(chunk)  # type: ignore[misc]
                used.add("gpu")
            except EngineError as exc:
                log.warning("GPU engine declined at %.1fs (%s) — finishing on CPU", chunk.start, exc)
                use_gpu = False
        if text is None:
            text = await cpu(chunk)
            used.add("cpu")
        text = text.strip()
        if not text:
            continue
        if is_degenerate(text):
            result.dropped.append({"start": round(chunk.start, 2), "end": round(chunk.end, 2)})
            log.warning("dropped a looping transcript for %.1f-%.1fs", chunk.start, chunk.end)
            continue
        result.segments.append({
            "id": len(result.segments), "start": round(chunk.start, 2), "end": round(chunk.end, 2), "text": text,
        })
    result.engine = "+".join(e for e in ("gpu", "cpu") if e in used) or "none"
    return result
