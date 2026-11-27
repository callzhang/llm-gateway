"""The real LiteLLM proxy, with this repo's litellm_settings, in front of a slow fake model_manager.

Cloudflare answers 524 when the origin sends nothing for ~100 s, and LiteLLM sends nothing until
model_manager's first chunk, so a streaming request held at the priority gate used to 524.
LiteLLM's TTFT keepalive (litellm_settings.sse_keepalive_ping_interval_seconds) opens the SSE
response and writes `: ping` comments meanwhile. These tests start the proxy the way
run_litellm.sh does (callbacks loaded from files next to the config) and check, on the installed
LiteLLM, that streaming requests get headers and pings before the first upstream byte, that
non-streaming requests are unchanged, and that request_priority_hook's priority reaches
model_manager as a top-level body field.
"""
from __future__ import annotations

import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LITELLM = os.path.join(os.path.dirname(sys.executable), "litellm")
MASTER_KEY = "sk-test-proxy-stream"
UPSTREAM_DELAY = 3.0
INTERVAL = 1.0

FAKE_UPSTREAM = textwrap.dedent(
    """
    import asyncio, json, sys, time
    from aiohttp import web

    async def chat(request):
        body = await request.json()
        with open(sys.argv[2], "a") as f:
            f.write(json.dumps(body) + "\\n")
        await asyncio.sleep(float(sys.argv[3]))
        model, now = body["model"], int(time.time())
        if not body.get("stream"):
            return web.json_response({"id": "c", "object": "chat.completion", "created": now, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)
        for delta, finish in (({"role": "assistant", "content": "hello"}, None), ({}, "stop")):
            chunk = {"id": "c", "object": "chat.completion.chunk", "created": now, "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
            await resp.write(f"data: {json.dumps(chunk)}\\n\\n".encode())
        await resp.write(b"data: [DONE]\\n\\n")
        await resp.write_eof()
        return resp

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host="127.0.0.1", port=int(sys.argv[1]), print=None)
    """
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_up(port: int, path: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            conn.request("GET", path)
            if conn.getresponse().status < 500:
                return
        except OSError:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"nothing answered on :{port}{path}")


def post(port: int, path: str, payload: dict) -> list[tuple[float, bytes]]:
    """POST and return (seconds since sending, bytes) for the headers and every read."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    started = time.monotonic()
    conn.request("POST", path, body=json.dumps(payload),
                 headers={"Authorization": f"Bearer {MASTER_KEY}", "Content-Type": "application/json"})
    response = conn.getresponse()
    events = [(time.monotonic() - started, f"HTTP {response.status}".encode())]
    while chunk := response.read1(65536):
        events.append((time.monotonic() - started, chunk))
    conn.close()
    return events


@unittest.skipUnless(os.path.exists(LITELLM), "needs the gateway env's litellm")
class LiteLLMProxyStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="litellm-proxy-stream-")
        repo_cfg = yaml.safe_load(open(os.path.join(REPO, "config.yaml")))
        cls.configured_interval = repo_cfg["litellm_settings"]["sse_keepalive_ping_interval_seconds"]
        # Callbacks are loaded from files next to the config, as on the host.
        for name in repo_cfg["litellm_settings"]["callbacks"]:
            shutil.copy(os.path.join(REPO, name.split(".")[0] + ".py"), cls.tmp)
        upstream_port, cls.proxy_port = free_port(), free_port()
        cls.bodies = os.path.join(cls.tmp, "bodies.jsonl")
        settings = {**repo_cfg["litellm_settings"], "sse_keepalive_ping_interval_seconds": INTERVAL}
        config = {
            "model_list": [{"model_name": "qwen3.8-27b", "litellm_params": {
                "model": "custom_openai/qwen3.8-27b", "api_base": f"http://127.0.0.1:{upstream_port}/v1",
                "api_key": "x", "allowed_openai_params": ["reasoning_effort"]}}],
            "router_settings": {"num_retries": 0, "disable_cooldowns": True},
            "litellm_settings": settings,
            "general_settings": {"master_key": MASTER_KEY},
        }
        config_path = os.path.join(cls.tmp, "config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(config, f)
        with open(os.path.join(cls.tmp, "fake_upstream.py"), "w") as f:
            f.write(FAKE_UPSTREAM)
        env = {k: v for k, v in os.environ.items() if k not in ("DATABASE_URL", "PYTHONPATH", "LITELLM_MASTER_KEY")}
        env.update(PYTHONNOUSERSITE="1", LITELLM_CONFIG="config.yaml")
        cls.procs = [
            subprocess.Popen([sys.executable, "fake_upstream.py", str(upstream_port), cls.bodies, str(UPSTREAM_DELAY)],
                             cwd=cls.tmp, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
            subprocess.Popen([LITELLM, "--config", config_path, "--port", str(cls.proxy_port), "--host", "127.0.0.1"],
                             cwd=cls.tmp, env=env, stdout=open(os.path.join(cls.tmp, "litellm.log"), "w"),
                             stderr=subprocess.STDOUT),
        ]
        try:
            wait_until_up(cls.proxy_port, "/health/liveliness", time.monotonic() + 90)
        except Exception:
            cls.tearDownClass()
            raise

    @classmethod
    def tearDownClass(cls) -> None:
        for proc in cls.procs:
            proc.terminate()
        for proc in cls.procs:
            proc.wait(10)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def last_body(self) -> dict:
        with open(self.bodies) as f:
            return json.loads(f.readlines()[-1])

    def test_the_configured_interval_answers_well_inside_cloudflares_100_seconds(self) -> None:
        self.assertGreater(self.configured_interval, 0)
        self.assertLessEqual(self.configured_interval, 20)

    def assert_pings_then_stream(self, events: list[tuple[float, bytes]], first_event: bytes) -> None:
        (header_at, status), (first_at, first) = events[0], events[1]
        self.assertEqual(b"HTTP 200", status)
        self.assertLess(header_at, UPSTREAM_DELAY - 0.5, "headers must not wait for the first upstream chunk")
        self.assertEqual(b": ping\n\n", first)
        self.assertLess(first_at, UPSTREAM_DELAY - 0.5)
        body = b"".join(chunk for _, chunk in events[1:])
        self.assertIn(first_event, body)
        self.assertTrue(body.rstrip().endswith(b"data: [DONE]"))

    def test_streaming_responses_get_headers_and_pings_before_the_first_token(self) -> None:
        events = post(self.proxy_port, "/v1/responses", {
            "model": "qwen3.8-27b", "input": "hi", "stream": True, "reasoning": {"effort": "low"},
            "extra_body": {"thinking_token_budget": 4096, "priority": 0}})
        self.assert_pings_then_stream(events, b'"type":"response.completed"')
        body = self.last_body()
        self.assertEqual(4096, body["thinking_token_budget"])
        self.assertEqual(1, body["priority"], "the master key holds no grant, so 0 becomes 1")

    def test_streaming_chat_gets_headers_and_pings_before_the_first_token(self) -> None:
        events = post(self.proxy_port, "/v1/chat/completions", {
            "model": "qwen3.8-27b", "stream": True, "messages": [{"role": "user", "content": "hi"}],
            "extra_body": {"priority": 0}})
        self.assert_pings_then_stream(events, b'"content":"hello"')
        self.assertEqual(1, self.last_body()["priority"], "the master key holds no grant, so 0 becomes 1")

    def test_a_bogus_previous_response_id_is_a_real_410_not_an_in_stream_error(self) -> None:
        """previous_response_hook must decide before the first keepalive commits a 200."""
        events = post(self.proxy_port, "/v1/responses", {
            "model": "qwen3.8-27b", "input": "hi", "stream": True, "previous_response_id": "resp_bogus_123"})
        header_at, status = events[0]
        self.assertEqual(b"HTTP 410", status)
        self.assertLess(header_at, INTERVAL)
        body = b"".join(chunk for _, chunk in events[1:])
        self.assertNotIn(b": ping", body)
        self.assertIn(b"previous_response_id", body)

    def test_model_manager_gets_the_checked_priority_from_every_route_and_placement(self) -> None:
        messages = [{"role": "user", "content": "hi"}]
        routes = {
            "/v1/chat/completions": {"messages": messages},
            "/v1/messages": {"messages": messages, "max_tokens": 5},
            "/v1/completions": {"prompt": "hi"},
        }
        for path, body in routes.items():
            for asked, expected in ((0, 1), (200, 200)):
                for placement in ({"priority": asked}, {"extra_body": {"priority": asked}}):
                    # LiteLLM drops a top-level priority on /v1/completions before the hook's
                    # result is sent; model_manager then applies its default (100).
                    if path == "/v1/completions" and "priority" in placement:
                        continue
                    with self.subTest(path=path, placement=placement):
                        open(self.bodies, "w").close()
                        post(self.proxy_port, path, {"model": "qwen3.8-27b", **body, **placement})
                        self.assertEqual(expected, self.last_body().get("priority"))

    def test_non_streaming_requests_wait_for_the_whole_answer_as_before(self) -> None:
        events = post(self.proxy_port, "/v1/chat/completions", {
            "model": "qwen3.8-27b", "messages": [{"role": "user", "content": "hi"}]})
        header_at, status = events[0]
        self.assertEqual(b"HTTP 200", status)
        self.assertGreaterEqual(header_at, UPSTREAM_DELAY - 0.2)
        self.assertNotIn(b": ping", b"".join(chunk for _, chunk in events[1:]))
        self.assertNotIn("priority", self.last_body())


if __name__ == "__main__":
    unittest.main()
