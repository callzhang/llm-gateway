from __future__ import annotations

import asyncio
import json
import logging
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tts_access_gateway.app import create_app
from tts_access_gateway.auth import AccessDenied, AccessPrincipal
from tts_access_gateway.config import GatewayConfig
from tts_access_gateway.policy import MODEL


class FakeVerifier:
    def __init__(self, *, reject: bool = False) -> None:
        self.reject = reject
        self.tokens: list[str] = []

    async def verify(self, token: str) -> AccessPrincipal:
        self.tokens.append(token)
        if self.reject or not token:
            raise AccessDenied("invalid Access JWT")
        return AccessPrincipal(
            kind="employee",
            actor="person@stardust.ai",
            subject="user-1",
        )


class GatewayAppTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.upstream_requests: list[dict[str, object]] = []

        async def upstream_speech(request: web.Request) -> web.Response:
            self.upstream_requests.append(
                {
                    "headers": dict(request.headers),
                    "payload": await request.json(),
                }
            )
            return web.Response(
                body=b"ID3-test-mp3",
                content_type="audio/mpeg",
            )

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/audio/speech", upstream_speech)
        self.upstream_server = TestServer(upstream_app)
        await self.upstream_server.start_server()
        self.verifier = FakeVerifier()
        self.config = GatewayConfig(
            team_domain="https://stardust.cloudflareaccess.com",
            policy_audience="policy-aud",
            litellm_api_key="internal-tts-key",
            allowed_service_ids=frozenset(),
            litellm_base_url=str(self.upstream_server.make_url("")).rstrip("/"),
        )
        self.client = TestClient(
            TestServer(create_app(self.config, verifier=self.verifier))
        )
        await self.client.start_server()

    async def asyncTearDown(self) -> None:
        await self.client.close()
        await self.upstream_server.close()

    async def test_proxy_strips_client_auth_and_streams_mp3(self):
        response = await self.client.post(
            "/v1/audio/speech",
            headers={
                "Cf-Access-Jwt-Assertion": "signed-access-jwt",
                "Authorization": "Bearer employee-oauth-token",
                "X-Forwarded-For": "spoofed",
            },
            json={
                "model": MODEL,
                "input": "hello",
                "voice": "Vivian",
                "instructions": "warm",
                "response_format": "mp3",
            },
        )

        self.assertEqual(200, response.status)
        self.assertEqual("audio/mpeg", response.headers["Content-Type"])
        self.assertEqual(b"ID3-test-mp3", await response.read())
        self.assertEqual(["signed-access-jwt"], self.verifier.tokens)
        upstream = self.upstream_requests[0]
        headers = upstream["headers"]
        self.assertEqual(
            "Bearer internal-tts-key",
            headers["Authorization"],
        )
        self.assertNotIn("Cf-Access-Jwt-Assertion", headers)
        self.assertNotEqual("spoofed", headers.get("X-Forwarded-For"))

    async def test_auth_route_json_and_model_listing_fail_closed(self):
        missing = await self.client.post(
            "/v1/audio/speech",
            json={
                "model": MODEL,
                "input": "hello",
                "voice": "Vivian",
            },
        )
        self.assertEqual(403, missing.status)

        chat = await self.client.post(
            "/v1/chat/completions",
            headers={"Cf-Access-Jwt-Assertion": "signed"},
            json={"model": "qwen3.6-27b"},
        )
        self.assertEqual(403, chat.status)

        invalid_json = await self.client.post(
            "/v1/audio/speech",
            headers={
                "Cf-Access-Jwt-Assertion": "signed",
                "Content-Type": "application/json",
            },
            data="{",
        )
        self.assertEqual(400, invalid_json.status)

        models = await self.client.get(
            "/v1/models",
            headers={"Cf-Access-Jwt-Assertion": "signed"},
        )
        self.assertEqual(200, models.status)
        payload = await models.json()
        self.assertEqual([MODEL], [item["id"] for item in payload["data"]])

    async def test_body_size_is_bounded(self):
        response = await self.client.post(
            "/v1/audio/speech",
            headers={"Cf-Access-Jwt-Assertion": "signed"},
            data=b"x" * (64 * 1024 + 1),
        )
        self.assertEqual(413, response.status)

    async def test_audit_log_omits_text_instruction_token_and_audio(self):
        records: list[str] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(self.format(record))

        logger = logging.getLogger("tts_access_gateway.audit")
        handler = Capture()
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        try:
            response = await self.client.post(
                "/v1/audio/speech",
                headers={
                    "Cf-Access-Jwt-Assertion": "private-jwt",
                    "Authorization": "Bearer private-oauth",
                },
                json={
                    "model": MODEL,
                    "input": "sensitive spoken text",
                    "voice": "Vivian",
                    "instructions": "secret delivery instruction",
                },
            )
            await response.read()
        finally:
            logger.removeHandler(handler)

        audit = "\n".join(records)
        parsed = json.loads(records[-1])
        self.assertEqual("person@stardust.ai", parsed["actor"])
        self.assertNotIn("sensitive spoken text", audit)
        self.assertNotIn("secret delivery instruction", audit)
        self.assertNotIn("private-jwt", audit)
        self.assertNotIn("private-oauth", audit)
        self.assertNotIn("ID3-test-mp3", audit)


class _CaptureAudit(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


class _RaisingSession:
    """Stand-in upstream session whose every call fails the way LiteLLM does."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc

    def post(self, *args, **kwargs):
        raise self.exc

    async def close(self) -> None:
        return None


class UpstreamFailureTests(unittest.IsolatedAsyncioTestCase):
    """A LiteLLM restart must not break the JSON error contract.

    Before this, an unreachable upstream escaped the middleware and aiohttp
    answered a public endpoint with its default HTML 500, unaudited.
    """

    async def asyncSetUp(self) -> None:
        self.config = GatewayConfig(
            team_domain="https://stardust.cloudflareaccess.com",
            policy_audience="policy-aud",
            litellm_api_key="internal-tts-key",
            allowed_service_ids=frozenset(),
            litellm_base_url="http://127.0.0.1:9",
        )
        self.app = create_app(self.config, verifier=FakeVerifier())
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()
        self.handler = _CaptureAudit()
        self.logger = logging.getLogger("tts_access_gateway.audit")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    async def asyncTearDown(self) -> None:
        self.logger.removeHandler(self.handler)
        await self.client.close()

    async def _speak(self):
        return await self.client.post(
            "/v1/audio/speech",
            headers={"Cf-Access-Jwt-Assertion": "signed"},
            json={"model": MODEL, "input": "hello", "voice": "Vivian"},
        )

    async def test_unreachable_upstream_returns_json_502(self):
        response = await self._speak()
        self.assertEqual(502, response.status)
        self.assertEqual("application/json", response.content_type)
        body = await response.json()
        self.assertEqual("access_error", body["error"]["type"])
        self.assertEqual("upstream unavailable", body["error"]["message"])
        audit = json.loads(self.handler.records[-1])
        self.assertEqual(502, audit["status"])
        self.assertIn("upstream unavailable", audit["reason"])

    async def test_upstream_timeout_returns_json_504(self):
        self.app["upstream_session"] = _RaisingSession(asyncio.TimeoutError())
        response = await self._speak()
        self.assertEqual(504, response.status)
        body = await response.json()
        self.assertEqual("upstream timed out", body["error"]["message"])
        audit = json.loads(self.handler.records[-1])
        self.assertEqual(504, audit["status"])


class DenialAuditTests(unittest.IsolatedAsyncioTestCase):
    """Every terminal response is audited, denials included.

    The audit logger is the only record — `access_log` is None — so an
    unaudited 403 means a probe against the public hostname leaves no trace.
    """

    async def asyncSetUp(self) -> None:
        config = GatewayConfig(
            team_domain="https://stardust.cloudflareaccess.com",
            policy_audience="policy-aud",
            litellm_api_key="internal-tts-key",
            allowed_service_ids=frozenset(),
            litellm_base_url="http://127.0.0.1:9",
        )
        self.client = TestClient(
            TestServer(create_app(config, verifier=FakeVerifier(reject=True)))
        )
        await self.client.start_server()
        self.handler = _CaptureAudit()
        self.logger = logging.getLogger("tts_access_gateway.audit")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.INFO)

    async def asyncTearDown(self) -> None:
        self.logger.removeHandler(self.handler)
        await self.client.close()

    async def test_rejected_identity_is_audited_as_unauthenticated(self):
        response = await self.client.post(
            "/v1/audio/speech",
            headers={"Cf-Access-Jwt-Assertion": "forged-jwt"},
            json={"model": MODEL, "input": "hello", "voice": "Vivian"},
        )
        self.assertEqual(403, response.status)
        audit = json.loads(self.handler.records[-1])
        self.assertEqual("unauthenticated", audit["actor_kind"])
        self.assertIsNone(audit["actor"])
        self.assertEqual(403, audit["status"])
        self.assertEqual("/v1/audio/speech", audit["route"])
        self.assertIn("reason", audit)
        self.assertNotIn("forged-jwt", "\n".join(self.handler.records))

    async def test_disallowed_route_is_audited_with_policy_reason(self):
        response = await self.client.post(
            "/v1/chat/completions",
            headers={"Cf-Access-Jwt-Assertion": "signed"},
            json={"model": "qwen3.6-27b"},
        )
        self.assertEqual(403, response.status)
        audit = json.loads(self.handler.records[-1])
        self.assertEqual("route not allowed", audit["reason"])
        self.assertEqual("unauthenticated", audit["actor_kind"])


if __name__ == "__main__":
    unittest.main()
