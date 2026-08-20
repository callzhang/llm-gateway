"""Authenticated, TTS-only streaming reverse proxy."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Mapping

from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    TCPConnector,
    web,
)

from .auth import AccessDenied, AccessJWTVerifier, AccessPrincipal
from .config import GatewayConfig
from .policy import (
    PolicyDenied,
    model_listing,
    validate_route,
    validate_speech_payload,
)


MAX_BODY_SIZE = 64 * 1024
AUDIT_LOGGER = logging.getLogger("tts_access_gateway.audit")


def _json_error(status: int, message: str, request_id: str) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": "access_error",
            },
            "request_id": request_id,
        },
        status=status,
        headers={"X-Request-Id": request_id},
    )


def _audit(request: web.Request, *, status: int, reason: str | None) -> None:
    """Emit exactly one audit record per terminal response.

    Denied requests have no verified principal, so they are recorded as an
    unauthenticated actor plus the denial reason.  Reasons come from our own
    policy and auth messages — never from request content — so the design's
    privacy rule (no input text, instructions, tokens, or audio in logs) holds
    for failures as well as successes.
    """
    principal: AccessPrincipal | None = request.get("principal")
    record = {
        "request_id": request.get("request_id"),
        "actor": principal.actor if principal is not None else None,
        "actor_kind": (
            principal.kind if principal is not None else "unauthenticated"
        ),
        "route": request.path,
        "model": request.get("audit_model"),
        "voice": request.get("audit_voice"),
        "status": status,
        "latency_ms": round(
            (time.monotonic() - request["started"]) * 1000, 2
        ),
        "output_bytes": request.get("output_bytes", 0),
    }
    if reason is not None:
        record["reason"] = reason
    AUDIT_LOGGER.info(
        json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    )


@web.middleware
async def error_middleware(
    request: web.Request,
    handler,
) -> web.StreamResponse:
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request["request_id"] = request_id
    request["started"] = time.monotonic()
    request["output_bytes"] = 0
    try:
        response = await handler(request)
    except AccessDenied as exc:
        return _denied(request, 403, "access denied", str(exc))
    except PolicyDenied as exc:
        return _denied(request, 403, str(exc), str(exc))
    except web.HTTPRequestEntityTooLarge:
        return _denied(request, 413, "request body too large", "body too large")
    except (json.JSONDecodeError, web.HTTPBadRequest, web.HTTPUnsupportedMediaType):
        return _denied(request, 400, "invalid JSON body", "invalid JSON body")
    except asyncio.TimeoutError as exc:
        # LiteLLM or the model took longer than the upstream budget.  A cold
        # Qwen3-TTS start is ~56s, so this is a real timeout, not a slow start.
        return _upstream_failure(request, 504, "upstream timed out", exc)
    except ClientError as exc:
        # LiteLLM restarts are routine on this host.  Without this the public
        # endpoint answered with aiohttp's default HTML 500 and logged nothing.
        return _upstream_failure(request, 502, "upstream unavailable", exc)
    _audit(request, status=response.status, reason=None)
    return response


def _denied(
    request: web.Request,
    status: int,
    message: str,
    reason: str,
) -> web.Response:
    _audit(request, status=status, reason=reason)
    return _json_error(status, message, request["request_id"])


def _upstream_failure(
    request: web.Request,
    status: int,
    message: str,
    exc: BaseException,
) -> web.Response:
    if request.get("prepared"):
        # The status line and part of the audio are already on the wire; the
        # 200 cannot be retracted.  Record the truncation and let aiohttp abort
        # the connection so the client sees a short read rather than a valid
        # but silently incomplete MP3.
        _audit(request, status=500, reason=f"{message} mid-stream")
        raise exc
    _audit(request, status=status, reason=f"{message}: {type(exc).__name__}")
    return _json_error(status, message, request["request_id"])


async def _handle(request: web.Request) -> web.StreamResponse:
    validate_route(request.method, request.path)
    if (
        request.content_length is not None
        and request.content_length > MAX_BODY_SIZE
    ):
        raise web.HTTPRequestEntityTooLarge(
            max_size=MAX_BODY_SIZE,
            actual_size=request.content_length,
        )

    verifier: AccessJWTVerifier = request.app["verifier"]
    request["principal"] = await verifier.verify(
        request.headers.get("Cf-Access-Jwt-Assertion", "")
    )
    request_id = request["request_id"]

    if request.path == "/v1/models":
        body = model_listing()
        request["output_bytes"] = len(
            json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        return web.json_response(
            body,
            headers={"X-Request-Id": request_id},
        )

    payload = await request.json()
    if not isinstance(payload, Mapping):
        raise PolicyDenied("JSON body must be an object")
    normalized = validate_speech_payload(payload)
    request["audit_model"] = str(normalized["model"])
    request["audit_voice"] = str(normalized["voice"])

    config: GatewayConfig = request.app["config"]
    session: ClientSession = request.app["upstream_session"]
    upstream_url = f"{config.litellm_base_url}/v1/audio/speech"
    # Built fresh rather than copied from the client request: no inbound
    # Authorization, Cloudflare identity, forwarding, or hop-by-hop header can
    # reach LiteLLM.
    upstream_headers = {
        "Authorization": f"Bearer {config.litellm_api_key}",
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
    }
    async with session.post(
        upstream_url,
        json=normalized,
        headers=upstream_headers,
    ) as source:
        response = web.StreamResponse(
            status=source.status,
            headers={
                "Content-Type": source.headers.get(
                    "Content-Type",
                    "application/json",
                ),
                "X-Request-Id": request_id,
            },
        )
        await response.prepare(request)
        request["prepared"] = True
        async for chunk in source.content.iter_chunked(64 * 1024):
            request["output_bytes"] += len(chunk)
            await response.write(chunk)
        await response.write_eof()

    return response


def create_app(
    config: GatewayConfig,
    *,
    verifier: AccessJWTVerifier | None = None,
) -> web.Application:
    app = web.Application(
        middlewares=[error_middleware],
        client_max_size=MAX_BODY_SIZE,
    )
    app["config"] = config
    app["verifier"] = verifier or AccessJWTVerifier(
        team_domain=config.team_domain,
        policy_audience=config.policy_audience,
        allowed_service_ids=config.allowed_service_ids,
    )

    async def create_session(application: web.Application) -> None:
        application["upstream_session"] = ClientSession(
            timeout=ClientTimeout(total=900),
            connector=TCPConnector(limit=16),
        )

    async def close_session(application: web.Application) -> None:
        await application["upstream_session"].close()

    app.on_startup.append(create_session)
    app.on_cleanup.append(close_session)
    app.router.add_route("*", "/{tail:.*}", _handle)
    return app
