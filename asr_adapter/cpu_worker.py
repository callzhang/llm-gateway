"""CPU transcription worker: Qwen3-ASR-0.6B (transformers, bf16) in its own process.

Started and stopped by model_manager (asr_adapter.cpu_pool); never run by hand except
to debug.  Protocol over stdin/stdout, so there is no port and no auth surface:

    parent -> worker   one JSON line {"samples": N, "language": str|null, "max_new_tokens": K}
                       followed by N little-endian float32 samples (16 kHz mono)
    worker -> parent   one JSON line {"ready": true} once after loading, then per request
                       {"text": "..."} or {"error": "..."}

EOF on stdin ends the worker, so it never outlives the manager that spawned it.
Stdout is reserved for the protocol: fd 1 is re-pointed at stderr for library output.
"""

from __future__ import annotations

import json
import os
import sys
import time

MODEL_ID = os.environ.get("ASR_CPU_MODEL", "Qwen/Qwen3-ASR-0.6B-hf")
THREADS = int(os.environ.get("ASR_CPU_THREADS", "8"))


def _read_exact(stream, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        piece = stream.read(n - len(buf))
        if not piece:
            raise EOFError("stdin closed mid-request")
        buf += piece
    return bytes(buf)


def main() -> int:
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)                                   # stray prints go to stderr, not the protocol

    def send(obj: dict) -> None:
        proto.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")

    try:
        os.nice(10)                                 # never starve the LLM pipeline
    except OSError:
        pass
    import numpy as np
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    torch.set_num_threads(THREADS)
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).eval()
    print(f"asr cpu worker: loaded {MODEL_ID} ({THREADS} threads) in {time.monotonic() - started:.1f}s",
          file=sys.stderr, flush=True)
    send({"ready": True, "model": MODEL_ID})

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            return 0
        try:
            req = json.loads(line)
            samples = np.frombuffer(_read_exact(stdin, int(req["samples"]) * 4), dtype="<f4")
            inputs = processor.apply_transcription_request(samples, language=req.get("language"))
            inputs = inputs.to(model.device, torch.bfloat16)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=int(req["max_new_tokens"]))
            text = processor.decode(ids[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")[0]
            send({"text": text})
        except EOFError:
            return 0
        except Exception as exc:                    # report, keep serving
            send({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    raise SystemExit(main())
