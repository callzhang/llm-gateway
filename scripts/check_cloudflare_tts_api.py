#!/usr/bin/env python3
"""Smoke-check Cloudflare AI model availability and TTS inference.

The script authenticates using the same Cloudflare token format as `wrangler` and
validates that compressed audio bytes can be produced for a given speech model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import quote


CF_API_ROOT = "https://api.cloudflare.com/client/v4"


def _token_from_env() -> str:
    token = (
        os.getenv("CLOUDFLARE_API_TOKEN")
        or os.getenv("CF_API_TOKEN")
        or os.getenv("CLOUDFLARE_TOKEN")
    )
    if not token:
        raise SystemExit(
            "Missing token: set one of CLOUDFLARE_API_TOKEN, CF_API_TOKEN, "
            "or CLOUDFLARE_TOKEN"
        )
    return token


def request_json(url: str, token: str) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        payload = ""
        try:
            payload = exc.read().decode("utf-8", errors="replace")
        except Exception:
            payload = str(exc)
        raise SystemExit(f"HTTP {exc.code} for {url}: {payload}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error for {url}: {exc.reason}")


def request_bytes(url: str, token: str, payload: dict[str, Any]) -> tuple[dict[str, str], bytes]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            headers = dict(resp.headers)
            return headers, resp.read()
    except urllib.error.HTTPError as exc:
        payload = ""
        try:
            payload = exc.read().decode("utf-8", errors="replace")
        except Exception:
            payload = str(exc)
        raise SystemExit(f"HTTP {exc.code} for inference: {payload}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"Network error for inference: {exc.reason}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check Cloudflare AI TTS inference via account token"
    )
    parser.add_argument("--account-id", required=True, help="Cloudflare account id")
    parser.add_argument(
        "--model",
        default="@cf/deepgram/aura-2-en",
        help="Model to run for inference smoke check",
    )
    parser.add_argument(
        "--text", default="开发环境测试", help="Input text for TTS inference"
    )
    parser.add_argument(
        "--voice", default="en-US", help="Optional model voice field, if supported"
    )
    parser.add_argument(
        "--response-format",
        default="mp3",
        choices=("mp3",),
        help="Expected response format",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/stardust-tts-smoke.mp3"),
        help="Output file for the synthesized audio",
    )
    return parser.parse_args()


def is_mp3(data: bytes) -> bool:
    return data.startswith(b"ID3") or (
        len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0
    )


def main() -> int:
    args = parse_args()
    token = _token_from_env()
    base = f"{CF_API_ROOT}/accounts/{args.account_id}"

    models = request_json(f"{base}/ai/models", token)
    if not models.get("success"):
        raise SystemExit("Cloudflare AI models API returned failure")

    run_url = f"{base}/ai/run/{quote(args.model, safe='')}"
    payload = {
        "text": args.text,
        "response_format": args.response_format,
        "voice": args.voice,
    }

    headers, audio = request_bytes(run_url, token, payload)
    content_type = (headers.get("Content-Type") or "").split(";", 1)[0]
    if content_type != "audio/mpeg":
        raise SystemExit(f"Unexpected content-type {content_type or 'missing'}")
    if not audio:
        raise SystemExit("Inference returned empty response body")
    if not is_mp3(audio):
        raise SystemExit("Response body is not MP3-encoded audio")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(audio)
    print(
        json.dumps(
            {
                "ok": True,
                "account_id": args.account_id,
                "model": args.model,
                "out": str(args.out),
                "bytes": len(audio),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
