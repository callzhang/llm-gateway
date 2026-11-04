"""model_manager fits chat requests into the model's context window before vLLM.

Functional tests drive DynamicRouter.handle with a stub tokenizer (1 token per
content character + 4 per message) and capture the body the backend receives.
The integration test uses the real tokenizer and config.yaml with a request
shaped like LiteLLM's Responses bridge output for a long previous_response_id
chain: all history already inside `messages`.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

import yaml
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import model_manager as mm
from context_fit import ContextFitter

MODEL = "qwen3.8-27b"
CTX = 10_000
CONFIG = {"model_list": [{
    "model_name": MODEL,
    "litellm_params": {"model": f"custom_openai/{MODEL}", "max_tokens": 3000},
    "model_info": {"max_input_tokens": CTX, "tokenizer": "stub/tok"},
}]}


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True, return_dict=True):
        n = sum(len(m.get("content") or "") + 4 for m in messages)
        return {"input_ids": [0] * n}


def turn(i, size):
    return [{"role": "user", "content": f"u{i}".ljust(size, "x")}, {"role": "assistant", "content": f"a{i}".ljust(size, "y")}]


class RouterHarness(unittest.IsolatedAsyncioTestCase):
    fitter: ContextFitter

    def make_fitter(self) -> ContextFitter:
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yaml")
            with open(path, "w") as f:
                yaml.safe_dump(CONFIG, f)
            return ContextFitter.from_config(path, tokenizer_loader=lambda repo, rev: StubTokenizer())

    async def asyncSetUp(self):
        config = mm.ModelConfig(script="unused.sh", served_name=MODEL, max_num_seqs=8, request_kind="chat")
        self.router = mm.DynamicRouter([mm.GpuSlot(0, 0, 9000)], {MODEL: config}, self.make_fitter())
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


class ContextFitInRouterTests(RouterHarness):
    async def test_a_request_that_fits_reaches_vllm_byte_for_byte(self):
        payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 500}
        raw = json.dumps(payload).encode()
        response = await self.client.post("/v1/chat/completions", data=raw, headers={"Content-Type": "application/json"})
        self.assertEqual(200, response.status)
        self.assertEqual([raw], self.forwarded)

    async def test_an_overflowing_history_is_trimmed_before_it_reaches_vllm(self):
        history = [m for i in range(6) for m in turn(i, 1000)]
        last = {"role": "user", "content": "last question"}
        payload = {"model": MODEL, "messages": history + [last], "max_tokens": 3000}

        response = await self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(200, response.status)
        sent = json.loads(self.forwarded[0])
        self.assertEqual(last, sent["messages"][-1])
        self.assertLess(len(sent["messages"]), len(payload["messages"]))
        tokens = StubTokenizer().apply_chat_template(sent["messages"])["input_ids"].__len__()
        self.assertLessEqual(tokens + sent["max_tokens"], CTX)

    async def test_the_sticky_key_is_taken_from_the_untrimmed_messages(self):
        """Dropping the oldest turns changes the first user message, which the
        sticky key hashes; the key must come from what the client sent."""
        history = [m for i in range(6) for m in turn(i, 1000)]
        payload = {"model": MODEL, "messages": history + [{"role": "user", "content": "last"}], "max_tokens": 3000}
        seen: list[int] = []
        real = mm._sticky_slot_for

        def spy(parsed_body, headers):
            seen.append(len(parsed_body["messages"]))
            return real(parsed_body, headers)

        with mock.patch.object(mm, "_sticky_slot_for", spy):
            await self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual([len(payload["messages"])], seen)
        self.assertLess(len(json.loads(self.forwarded[0])["messages"]), seen[0])

    async def test_a_body_without_messages_is_forwarded_unchanged(self):
        raw = json.dumps({"model": MODEL, "prompt": "x" * 50_000, "max_tokens": 3000}).encode()
        await self.client.post("/v1/completions", data=raw, headers={"Content-Type": "application/json"})
        self.assertEqual([raw], self.forwarded)


class ContextFitWithRealTokenizerTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_long_previous_response_chain_is_made_to_fit(self):
        """What LiteLLM sends for previous_response_id after it appended the
        stored history: tool calls, results and old reasoning, ~170k tokens
        against a 131072 window."""
        fitter = ContextFitter.from_config()
        limits = fitter._limits[MODEL]
        filler = "alpha beta gamma delta epsilon zeta eta theta " * 400
        messages = [{"role": "system", "content": "You are a coding agent."}]
        for t in range(40):
            messages += [
                {"role": "user", "content": f"step {t}"},
                {"role": "assistant", "content": None, "reasoning_content": f"think {t} " + filler,
                 "tool_calls": [{"id": f"c{t}", "type": "function",
                                 "function": {"name": "read", "arguments": json.dumps({"path": f"f{t}"})}}]},
                {"role": "tool", "tool_call_id": f"c{t}", "content": f"file {t} " + filler * 3},
                {"role": "assistant", "content": f"done {t}"},
            ]
        messages.append({"role": "user", "content": "final step"})
        body = {"model": MODEL, "messages": messages, "max_tokens": 32768}
        before = limits.tokenizer.count(messages, None)
        self.assertGreater(before + 32768, limits.context_window)

        self.assertTrue(await fitter.fit(body, MODEL))
        after = limits.tokenizer.count(body["messages"], None)
        self.assertLessEqual(after + body["max_tokens"], limits.context_window)
        self.assertEqual({"role": "user", "content": "final step"}, body["messages"][-1])
        self.assertEqual("system", body["messages"][0]["role"])
        # every tool result still has the assistant tool call it answers
        calls = {c["id"] for m in body["messages"] for c in m.get("tool_calls") or []}
        results = {m["tool_call_id"] for m in body["messages"] if m["role"] == "tool"}
        self.assertEqual(results, results & calls)


if __name__ == "__main__":
    unittest.main()
