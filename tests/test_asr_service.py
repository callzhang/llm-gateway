import asyncio
import shutil
import subprocess

import numpy as np
import pytest

from asr_adapter import service
from asr_adapter.audio import SAMPLE_RATE
from asr_adapter.engine import EngineError
from asr_adapter.service import UploadError, Upload, parse_upload, render, transcribe_upload

BOUNDARY = "----testboundary"
CT = f"multipart/form-data; boundary={BOUNDARY}"


def form(fields, audio=b"RIFFdata", filename="a.wav"):
    parts = []
    for k, v in fields.items():
        parts.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    if audio is not None:
        parts.append(f'--{BOUNDARY}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                     f'Content-Type: audio/wav\r\n\r\n'.encode() + audio + b"\r\n")
    parts.append(f"--{BOUNDARY}--\r\n".encode())
    return b"".join(parts)


def test_parse_upload_reads_fields_and_file_bytes():
    audio = bytes(range(256)) + f"--{BOUNDARY}x".encode()      # binary, even looks like a delimiter
    up = parse_upload(form({"model": "qwen3-asr-0.6b", "language": "zh", "response_format": "verbose_json"}, audio), CT)
    assert (up.model, up.language, up.response_format) == ("qwen3-asr-0.6b", "zh", "verbose_json")
    assert up.audio == audio


def test_defaults_and_validation():
    assert parse_upload(form({}), CT).response_format == "json"
    with pytest.raises(UploadError, match="response_format"):
        parse_upload(form({"response_format": "srt"}), CT)
    with pytest.raises(UploadError, match="file"):
        parse_upload(form({"model": "m"}, audio=None), CT)
    with pytest.raises(UploadError, match="multipart"):
        parse_upload(b"{}", "application/json")


def test_oversized_upload_is_413(monkeypatch):
    monkeypatch.setattr(service, "MAX_UPLOAD_BYTES", 4)
    with pytest.raises(UploadError) as e:
        parse_upload(form({}, b"12345"), CT)
    assert e.value.status == 413


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_end_to_end_with_real_ffmpeg_decode_and_fake_engines():
    t = np.arange(SAMPLE_RATE * 3) / SAMPLE_RATE
    pcm = (0.2 * np.sin(2 * np.pi * 300 * t) * 32767).astype("<i2").tobytes()
    wav = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1", "-i", "pipe:0",
                          "-f", "wav", "pipe:1"], input=pcm, capture_output=True, check=True).stdout
    up = parse_upload(form({"response_format": "verbose_json", "language": "zh"}, wav), CT)

    async def gpu(chunk):
        raise EngineError("HTTP 503: gpu_busy")

    async def cpu(chunk):
        return "你好世界"

    duration, result = asyncio.run(transcribe_upload(up, gpu=gpu, cpu=cpu))
    assert duration == pytest.approx(3.0, abs=0.05) and result.engine == "cpu"
    kind, body = render(up, duration, result)
    assert kind == "json" and body["text"] == "你好世界" and body["engine"] == "cpu"
    assert body["segments"][0]["start"] == pytest.approx(0.0, abs=0.35)


def test_undecodable_audio_is_a_400():
    up = Upload(b"not audio at all", None, None, "json")
    with pytest.raises(UploadError) as e:
        asyncio.run(transcribe_upload(up, gpu=None, cpu=None))
    assert e.value.status == 400


def test_render_shapes():
    from asr_adapter.engine import TranscriptionResult
    res = TranscriptionResult(segments=[{"id": 0, "start": 0, "end": 1, "text": "hello"},
                                        {"id": 1, "start": 1, "end": 2, "text": "world"}], engine="gpu")
    assert render(Upload(b"", None, None, "text"), 2.0, res) == ("text", "hello world")
    assert render(Upload(b"", None, None, "json"), 2.0, res) == ("json", {"text": "hello world"})
