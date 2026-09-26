"""POST /v1/audio/transcriptions handling: parse the upload, decode, chunk, transcribe.

model_manager wires in the two engines; this module owns everything model-independent:
the multipart parse, ffmpeg decoding, chunking/silence/loop protection (audio.py) and
the OpenAI response shapes.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass

import numpy as np

from .audio import SAMPLE_RATE, join_text, make_chunks
from .engine import Engine, TranscriptionResult, transcribe_chunks

RESPONSE_FORMATS = {"json", "verbose_json", "text"}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_DURATION_SECONDS = 4 * 3600
# Target chunk length.  <=60 s fits the ASR lane's --max-model-len 1536 (audio ~12.5 tok/s
# plus the transcript); longer needs a larger max-model-len.
CHUNK_SECONDS = float(os.environ.get("ASR_CHUNK_SECONDS", "30"))


class UploadError(ValueError):
    """The request is not a usable transcription upload; `status` is the HTTP code."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Upload:
    audio: bytes
    model: str | None
    language: str | None
    response_format: str


def parse_upload(body: bytes, content_type: str) -> Upload:
    """The `file`, `model`, `language` and `response_format` fields of an OpenAI
    transcription form.  Splits on the boundary with bytes.find so a 100 MB upload is
    not copied line by line."""
    match = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', content_type, re.I)
    if not content_type.lower().startswith("multipart/form-data") or not match:
        raise UploadError("expected a multipart/form-data upload")
    dash = b"--" + (match.group(1) or match.group(2)).encode()
    sep = b"\r\n" + dash          # a delimiter only counts at the start of a line (RFC 2046)
    fields: dict[str, str] = {}
    audio: bytes | None = None
    pos = 0 if body.startswith(dash) else body.find(sep) + 2 if body.find(sep) != -1 else -1
    while pos != -1:
        start = pos + len(dash)
        if body[start:start + 2] == b"--":
            break
        nxt = body.find(sep, start)
        end = nxt if nxt != -1 else len(body)
        head_end = body.find(b"\r\n\r\n", start, end)
        if head_end != -1:
            head = body[start:head_end].decode("utf-8", "replace")
            value = body[head_end + 4:end]
            name = re.search(r'\bname="([^"]*)"', head)
            if name and name.group(1) == "file":
                audio = value
            elif name:
                fields[name.group(1)] = value.decode("utf-8", "replace").strip()
        pos = nxt + 2 if nxt != -1 else -1
    if not audio:
        raise UploadError("missing or empty 'file' field")
    if len(audio) > MAX_UPLOAD_BYTES:
        raise UploadError(f"audio exceeds {MAX_UPLOAD_BYTES} bytes", 413)
    fmt = fields.get("response_format") or "json"
    if fmt not in RESPONSE_FORMATS:
        raise UploadError(f"unsupported response_format: {fmt}")
    return Upload(audio, fields.get("model") or None, fields.get("language") or None, fmt)


async def decode_audio(payload: bytes) -> np.ndarray:
    """16 kHz mono float32 samples of whatever container/codec ffmpeg understands."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "pipe",
        "-i", "pipe:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(payload)
    if proc.returncode != 0 or not out:
        raise UploadError(f"audio could not be decoded: {err.decode(errors='replace')[:200]}")
    return np.frombuffer(out, dtype=np.float32).copy()


async def transcribe_upload(
    upload: Upload, *, gpu: Engine | None, cpu: Engine,
) -> tuple[float, TranscriptionResult]:
    audio = await decode_audio(upload.audio)
    duration = round(len(audio) / SAMPLE_RATE, 2)
    if duration > MAX_DURATION_SECONDS:
        raise UploadError(f"audio is {duration:.0f}s; the limit is {MAX_DURATION_SECONDS}s", 413)
    chunks = await asyncio.to_thread(make_chunks, audio, target_s=CHUNK_SECONDS)
    return duration, await transcribe_chunks(chunks, gpu=gpu, cpu=cpu)


def render(upload: Upload, duration: float, result: TranscriptionResult) -> tuple[str, object]:
    """(kind, payload): "text" -> str body, "json" -> dict body, per the requested format."""
    text = join_text([s["text"] for s in result.segments])
    if upload.response_format == "text":
        return "text", text
    if upload.response_format == "json":
        return "json", {"text": text}
    return "json", {
        "task": "transcribe", "language": upload.language, "duration": duration, "text": text,
        "segments": result.segments, "dropped_segments": result.dropped, "engine": result.engine,
    }
