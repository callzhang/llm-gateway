"""Authenticated, TTS-only streaming reverse proxy."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Mapping

from aiohttp import ClientSession, ClientTimeout, TCPConnector, web

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


def _audit(
    *,
    principal: AccessPrincipal,
    request_id: str,
    route: str,
    model: str | None,
    voice: str | None,
    status: int,
    started: float,
    output_bytes: int,
) -> None:
    AUDIT_LOGGER.info(
        json.dumps(
            {
                "request_id": request_id,
                "actor": principal.actor,
                "actor_kind": principal.kind,
                "route": route,
                "model": model,
                "voice": voice,
                "status": status,
                "latency_ms": round((time.monotonic() - started) * 1000, 2),
                "output_bytes": output_bytes,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


@web.middleware
async def error_middleware(
    request: web.Request,
    handler,
) -> web.StreamResponse:
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    request["request_id"] = request_id
    try:
        return await handler(request)
    except AccessDenied:
        return _json_error(403, "access denied", request_id)
    except PolicyDenied as exc:
        return _json_error(403, str(exc), request_id)
    except web.HTTPRequestEntityTooLarge:
        return _json_error(413, "request body too large", request_id)
    except (json.JSONDecodeError, web.HTTPBadRequest, web.HTTPUnsupportedMediaType):
        return _json_error(400, "invalid JSON body", request_id)


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
    principal = await verifier.verify(
        request.headers.get("Cf-Access-Jwt-Assertion", "")
    )
    request_id = request["request_id"]
    started = time.monotonic()

    if request.path == "/v1/models":
        body = model_listing()
        encoded_size = len(
            json.dumps(body, ensure_ascii=False).encode("utf-8")
        )
        _audit(
            principal=principal,
            request_id=request_id,
            route=request.path,
            model=None,
            voice=None,
            status=200,
            started=started,
            output_bytes=encoded_size,
        )
        return web.json_response(
            body,
            headers={"X-Request-Id": request_id},
        )

    payload = await request.json()
    if not isinstance(payload, Mapping):
        raise PolicyDenied("JSON body must be an object")
    normalized = validate_speech_payload(payload)

    config: GatewayConfig = request.app["config"]
    session: ClientSession = request.app["upstream_session"]
    upstream_url = f"{config.litellm_base_url}/v1/audio/speech"
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
        output_bytes = 0
        async for chunk in source.content.iter_chunked(64 * 1024):
            output_bytes += len(chunk)
            await response.write(chunk)
        await response.write_eof()

    _audit(
        principal=principal,
        request_id=request_id,
        route=request.path,
        model=str(normalized["model"]),
        voice=str(normalized["voice"]),
        status=response.status,
        started=started,
        output_bytes=output_bytes,
    )
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
