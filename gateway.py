"""
LLM Gateway — FastAPI entry point sitting in front of LiteLLM.

Listens on :8900.
- GET  /health → aggregated health check
- Everything else → reverse-proxied to LiteLLM :8901
  (Responses API with previous_response_id handled natively by LiteLLM + Postgres)

OCR: call the RapidOCR service directly at https://ocr.preseen.ai/v1/ocr
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

LITELLM_BASE = os.environ.get("LITELLM_BASE",       "http://127.0.0.1:8901")
LITELLM_KEY  = os.environ.get("LITELLM_MASTER_KEY", "sk-local-gateway")

app = FastAPI(title="LLM Gateway", version="1.0.0")
_client = httpx.AsyncClient(timeout=600.0)


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
    global _client
    url = f"{LITELLM_BASE}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"

    body = await request.body()
    headers = {
        k: v for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    try:
        upstream = await _client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
        )
    except httpx.RemoteProtocolError:
        # Stale pooled connection (e.g. LiteLLM was restarted) — drop the
        # pool, build a fresh client, retry once.  Without this a LiteLLM
        # restart leaves the gateway returning 500s until it's restarted too.
        if not _client.is_closed:
            await _client.aclose()
        _client = httpx.AsyncClient(timeout=600.0)
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
