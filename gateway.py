"""
LLM Gateway — FastAPI entry point sitting in front of LiteLLM.

Listens on :8900.
- POST /v1/ocr  → handled here, routes to vision model
- Everything else → reverse-proxied to LiteLLM :8901
  (Responses API with previous_response_id handled natively by LiteLLM + Postgres)
"""
from __future__ import annotations

import os
import time
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

LITELLM_BASE  = os.environ.get("LITELLM_BASE",      "http://127.0.0.1:8901")
LITELLM_KEY   = os.environ.get("LITELLM_MASTER_KEY", "sk-local-gateway")
OCR_DEFAULT_MODEL   = os.environ.get("OCR_MODEL",     "qwen3.6-35b-a3b")
OCR_MODEL_FALLBACKS = os.environ.get("OCR_FALLBACKS", "gpt-5.4-mini").split(",")

OCR_DEFAULT_PROMPT = (
    "Extract ALL text from this image exactly as it appears. "
    "Preserve layout, line breaks, and punctuation. "
    "Output only the extracted text with no commentary."
)

app = FastAPI(title="LLM Gateway", version="1.0.0")
_client = httpx.AsyncClient(timeout=600.0)


# ── OCR endpoint ──────────────────────────────────────────────────────────────

@app.post("/v1/ocr")
async def ocr(request: Request) -> JSONResponse:
    """
    POST /v1/ocr
    {
      "image": "<base64>",          // raw base64 (no data: prefix needed)
      "url": "https://...",         // OR image URL
      "prompt": "optional prompt",
      "language": "zh",             // optional hint, appended to prompt
      "model": "qwen3.6-35b-a3b"   // optional — any vision model in the gateway
    }
    """
    body: dict[str, Any] = await request.json()

    model     = body.get("model", OCR_DEFAULT_MODEL)
    fallbacks = body.get("fallbacks", OCR_MODEL_FALLBACKS)
    prompt    = body.get("prompt", OCR_DEFAULT_PROMPT)
    if lang := body.get("language"):
        prompt += f" The text is primarily in {lang}."

    if url := body.get("url"):
        image_block = {"type": "image_url", "image_url": {"url": url}}
    elif raw := body.get("image"):
        if not raw.startswith("data:"):
            raw = f"data:image/jpeg;base64,{raw}"
        image_block = {"type": "image_url", "image_url": {"url": raw}}
    else:
        return JSONResponse({"error": "Provide 'image' (base64) or 'url'"}, status_code=400)

    payload = {
        "model": model,
        "fallbacks": fallbacks,
        "messages": [{"role": "user", "content": [image_block, {"type": "text", "text": prompt}]}],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=600.0) as c:
        r = await c.post(
            f"{LITELLM_BASE}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
        )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    if r.status_code != 200:
        return JSONResponse({"error": "OCR failed", "detail": r.text, "model": model},
                            status_code=r.status_code)

    data = r.json()
    text = data["choices"][0]["message"]["content"]
    return JSONResponse({
        "text": text,
        "model": data.get("model", model),
        "latency_ms": latency_ms,
        "usage": data.get("usage", {}),
    })


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    results: dict[str, Any] = {"gateway": "ok"}
    try:
        r = await _client.get(
            f"{LITELLM_BASE}/health",
            headers={"Authorization": f"Bearer {LITELLM_KEY}"},
            timeout=3.0,
        )
        results["litellm"] = r.json() if r.status_code == 200 else {"status": r.status_code}
    except Exception as e:
        results["litellm"] = {"error": str(e)}
    return JSONResponse(results)


# ── Reverse proxy: everything else → LiteLLM ─────────────────────────────────

@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy(request: Request, path: str) -> Response:
    url = f"{LITELLM_BASE}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    upstream = await _client.request(
        method=request.method,
        url=url,
        headers=headers,
        content=body,
    )

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=dict(upstream.headers),
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8900))
    uvicorn.run("gateway:app", host="0.0.0.0", port=port, reload=False)
