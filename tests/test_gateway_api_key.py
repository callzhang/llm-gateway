"""model_manager serves inference only to the gateway's own LiteLLMs (GATEWAY_API_KEY).

It listens on the tailnet for the peer region, and it puts the engine key on every forwarded request, so
without this check anyone who could connect used the models without a virtual key, limit or usage record.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import unittest
from pathlib import Path
from unittest import mock

import yaml

import model_manager as mm
from tests.test_model_manager_context_fit import MODEL, RouterHarness

ROOT = Path(__file__).resolve().parents[1]
KEY = "gateway-key-for-tests"
BODY = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}


class RouterTests(RouterHarness):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        patcher = mock.patch.object(mm, "GATEWAY_API_KEY", KEY)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_inference_needs_the_gateway_key(self):
        for headers in ({}, {"Authorization": "Bearer local-qwen36"}, {"Authorization": KEY},
                        {mm.OVERFLOW_HEADER: "1"}):
            for path in ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/anything"):
                response = await self.client.post(path, json=BODY, headers=headers)
                self.assertEqual(401, response.status, (path, headers))
        self.assertEqual([], self.forwarded)
        response = await self.client.post("/v1/chat/completions", json=BODY, headers={"Authorization": f"Bearer {KEY}"})
        self.assertEqual(200, response.status)
        self.assertEqual(1, len(self.forwarded))

    async def test_health_and_status_stay_open(self):
        for path in ("/health", "/v1/models", "/status.json", "/admin/status"):
            self.assertEqual(200, (await self.client.get(path)).status, path)


class StartupTests(unittest.TestCase):
    def test_the_manager_refuses_to_start_without_a_key(self):
        with mock.patch.object(mm, "ENGINE_API_KEY", "engine"), mock.patch.object(mm, "GATEWAY_API_KEY", ""), \
                self.assertRaisesRegex(SystemExit, "GATEWAY_API_KEY"):
            asyncio.run(mm.main())

    def test_every_litellm_deployment_sends_it(self):
        for name in ("config.yaml", "config.cn.yaml"):
            with open(ROOT / name) as f:
                entries = yaml.safe_load(f)["model_list"]
            self.assertEqual({"os.environ/GATEWAY_API_KEY"}, {e["litellm_params"]["api_key"] for e in entries}, name)
        self.assertIn("${GATEWAY_API_KEY:?", (ROOT / "run_litellm.sh").read_text())


if __name__ == "__main__":
    unittest.main()
