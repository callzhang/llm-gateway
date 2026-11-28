"""vLLM / vLLM-Omni ASGI middleware: every route needs ENGINE_API_KEY, bar the probes.

vLLM's own --api-key checks only a few path prefixes, which leaves its native routes
(/invocations, /tokenize, /scale_elastic_ep, ...) open.  Loaded with --middleware.
"""

import os
import secrets

# model_manager, monitor.py and warm_jit_cache.sh read these without a key.
OPEN_PATHS = frozenset({"/health", "/metrics", "/metrics/"})


class RequireApiKey:
    def __init__(self, app) -> None:
        self.app = app
        key = os.environ["ENGINE_API_KEY"]
        if not key:
            raise RuntimeError("ENGINE_API_KEY is empty")
        self._expected = f"Bearer {key}".encode()

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "lifespan" or (
            scope["type"] == "http" and scope["path"] in OPEN_PATHS
        ):
            return await self.app(scope, receive, send)
        given = dict(scope["headers"]).get(b"authorization", b"")
        if secrets.compare_digest(given, self._expected):
            return await self.app(scope, receive, send)
        if scope["type"] == "websocket":
            return await send({"type": "websocket.close", "code": 1008})
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b'{"error": "Unauthorized"}'})
