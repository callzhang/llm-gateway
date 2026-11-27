"""model_manager turns vLLM's thinking_token_budget into SGLang's thinking-budget processor.

SGLang ignores `thinking_token_budget`; without the translation an SGLang backend thinks without limit
(CN 2026-11-14 to 11-16: community summaries hit the 8,192-token output ceiling and 300 s timeouts).
"""
from __future__ import annotations

import json
import pickletools
import unittest

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import model_manager as mm

MODEL = "qwen3.8-27b"


class Harness(unittest.IsolatedAsyncioTestCase):
    engine = "sglang"

    async def asyncSetUp(self):
        config = mm.ModelConfig(script="unused.sh", served_name=MODEL, max_num_seqs=8, request_kind="chat", engine=self.engine)
        self.router = mm.DynamicRouter([mm.GpuSlot(0, 0, 9000)], {MODEL: config})
        self.forwarded: list[bytes] = []

        async def proxy(request, body, priority=mm.DEFAULT_REQUEST_PRIORITY):
            self.forwarded.append(body)
            return web.json_response({"ok": True})

        backend = type("B", (), {"slot": mm.GpuSlot(0, 0, 9000), "proxy": staticmethod(proxy)})()

        async def get_or_start(_model):
            return [backend]

        self.router._get_or_start = get_or_start
        self.router._pick = lambda backends: backends[0]
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", self.router.handle)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def post(self, body: dict):
        return await self.client.post("/v1/chat/completions", data=json.dumps(body), headers={"content-type": "application/json"})


class SglangBackend(Harness):
    async def test_the_budget_becomes_the_qwen38_thinking_budget_processor(self):
        resp = await self.post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                                "thinking_token_budget": 4096, "custom_params": {"other": 1}})
        self.assertEqual(resp.status, 200)
        sent = json.loads(self.forwarded[0])
        self.assertNotIn("thinking_token_budget", sent)
        self.assertEqual(sent["custom_params"], {"other": 1, "thinking_budget": 4096})
        self.assertEqual(sent["custom_logit_processor"], mm.SGLANG_QWEN38_THINKING_BUDGET_PROCESSOR)

    async def test_a_request_without_a_budget_is_forwarded_unchanged(self):
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
        await self.post(body)
        self.assertEqual(json.loads(self.forwarded[0]), body)

    async def test_an_invalid_budget_is_refused(self):
        for bad in (-1, "4096", 1.5, True):
            resp = await self.post({"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "thinking_token_budget": bad})
            self.assertEqual(resp.status, 400, bad)
        self.assertEqual(self.forwarded, [])


class VllmBackend(Harness):
    engine = "vllm"

    async def test_vllm_keeps_its_own_budget_parameter(self):
        body = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "thinking_token_budget": 4096}
        await self.post(body)
        self.assertEqual(json.loads(self.forwarded[0]), body)


class RequestsTheEngineMayNotSee(Harness):
    """Client fields an engine would turn into code or loader arguments, and routes it must not receive."""

    CHAT = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}
    FIELDS = (
        {"custom_logit_processor": "client-value"},
        {"custom_logit_processor": ["client-value"]},
        {"custom_logit_processor": "client-value", "thinking_token_budget": 16},
        {"mm_processor_kwargs": {"subfolder": "client-value"}},
        {"chat_template": "client-value"},
        {"chat_template_kwargs": {"chat_template": "client-value"}},
    )

    async def send(self, path: str, data: bytes, method: str = "POST", headers: dict | None = None):
        return await self.client.request(method, path, data=data,
                                         headers={"content-type": "application/json", **(headers or {})})

    async def test_the_fields_are_refused_on_every_served_route_and_from_the_peer_region(self):
        for path in ("/v1/chat/completions", "/v1/completions", "/v1/responses"):
            for headers in ({}, {mm.OVERFLOW_HEADER: "1"}):
                for extra in self.FIELDS:
                    with self.subTest(path=path, headers=headers, extra=extra):
                        resp = await self.send(path, json.dumps({**self.CHAT, **extra}).encode(), headers=headers)
                        self.assertEqual(resp.status, 400)
        self.assertEqual(self.forwarded, [])

    async def test_other_engine_routes_and_methods_are_not_forwarded(self):
        body = json.dumps(self.CHAT).encode()
        for path in ("/generate", "/invocations", "/vertex_generate", "/update_weights_from_tensor",
                     "/load_lora_adapter", "/v1/chat/completions/", "/chat/completions", "/v1/chat/completions%3F"):
            with self.subTest(path=path):
                self.assertEqual((await self.send(path, body)).status, 400)
        for method in ("GET", "PUT"):
            with self.subTest(method=method):
                self.assertEqual((await self.send("/v1/chat/completions", body, method=method)).status, 400)
        self.assertEqual(self.forwarded, [])

    async def test_a_body_that_is_not_a_json_object_is_refused(self):
        multipart = (b'--b\r\nContent-Disposition: form-data; name="model"\r\n\r\n' + MODEL.encode() + b"\r\n--b--\r\n")
        resp = await self.send("/v1/chat/completions", multipart, headers={"content-type": "multipart/form-data; boundary=b"})
        self.assertEqual(resp.status, 400)
        self.assertEqual(self.forwarded, [])

    async def test_the_engine_gets_what_was_checked_not_the_client_bytes(self):
        raw = json.dumps(self.CHAT)[:-1] + ', "custom_logit_processor": "client-value", "custom_logit_processor": null}'
        self.assertEqual((await self.send("/v1/chat/completions", raw.encode())).status, 200)
        self.assertNotIn(b"client-value", self.forwarded[0])

    async def test_ordinary_template_arguments_still_pass(self):
        body = {**self.CHAT, "chat_template_kwargs": {"enable_thinking": False}}
        await self.send("/v1/chat/completions", json.dumps(body).encode())
        self.assertEqual(json.loads(self.forwarded[0]), body)


class RequestsAVllmEngineMayNotSee(RequestsTheEngineMayNotSee):
    engine = "vllm"


class ProcessorReference(unittest.TestCase):
    def test_the_processor_string_only_names_the_plugin_class(self):
        payload = bytes.fromhex(json.loads(mm.SGLANG_QWEN38_THINKING_BUDGET_PROCESSOR)["callable"])
        ops = [(op.name, arg) for op, arg, _pos in pickletools.genops(payload)]
        self.assertEqual(
            [arg for name, arg in ops if name == "SHORT_BINUNICODE"],
            ["friday_thinking_budget", "Qwen38ThinkingBudgetLogitProcessor"],
        )
        self.assertEqual([name for name, _ in ops if name not in ("PROTO", "FRAME", "MEMOIZE", "SHORT_BINUNICODE")],
                         ["STACK_GLOBAL", "STOP"])

    def test_the_plugin_names_qwen38_think_tokens(self):
        source = open("sglang_plugins/friday_thinking_budget.py").read()
        self.assertIn("class Qwen38ThinkingBudgetLogitProcessor(CustomLogitProcessor)", source)
        for token_id in ("248068", "248069", "198"):
            self.assertIn(token_id, source)


if __name__ == "__main__":
    unittest.main()
