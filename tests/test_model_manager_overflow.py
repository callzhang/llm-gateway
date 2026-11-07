"""Cross-region overflow: a long local queue turns new work away with a 503.

Each region runs its own gateway stack.  LiteLLM lists the other region's
gateway as an order-2 deployment of the same model, so a 503 from this
model_manager is what moves a request across the border.  The rule here:

  * only chat work that has no sticky slot here is turned away — a sticky hit
    keeps its prefix cache local;
  * a request that already crossed (x-gateway-overflow header, set by the
    peer's LiteLLM deployment) is never turned away again, so the two regions
    cannot bounce a request between them;
  * OVERFLOW_QUEUE_DEPTH=0 (the default) disables the rule entirely.
"""

import json
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import model_manager as mm


class OverflowTests(unittest.IsolatedAsyncioTestCase):
    MODEL = "qwen3.8-27b"

    async def asyncSetUp(self):
        config = mm.ModelConfig(
            script="unused.sh", served_name=self.MODEL, max_num_seqs=8, request_kind="chat",
        )
        self.router = mm.DynamicRouter(
            [mm.GpuSlot(0, 0, 9000), mm.GpuSlot(1, 1, 9010)], {self.MODEL: config},
        )
        self.proxied: list[int] = []

        def backend_on(slot):
            async def proxy(request, body, priority=mm.DEFAULT_REQUEST_PRIORITY):
                self.proxied.append(slot.slot_id)
                return web.json_response({"ok": True})
            return type("B", (), {"slot": slot, "proxy": staticmethod(proxy)})()

        self.backends = [backend_on(s) for s in self.router.slots]

        async def get_or_start(_model):
            return self.backends

        self.router._get_or_start = get_or_start
        self.router._pick = lambda backends: backends[0]
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", self.router.handle)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def _post(self, headers=None):
        payload = {"model": self.MODEL, "messages": [{"role": "user", "content": "hi"}]}
        return await self.client.post(
            "/v1/chat/completions", json=payload, headers=headers or {},
        )

    async def test_long_queue_turns_new_work_away_with_a_503(self):
        self.router._queue_waiting[self.MODEL] = 4
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(503, response.status)
        self.assertEqual("queue_overflow", (await response.json())["error"]["type"])
        self.assertEqual([], self.proxied)

    async def test_queue_below_the_depth_is_served_locally(self):
        self.router._queue_waiting[self.MODEL] = 3
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(200, response.status)
        self.assertEqual([0], self.proxied)

    async def test_disabled_by_default(self):
        self.router._queue_waiting[self.MODEL] = 1000
        self.assertEqual(0, mm.OVERFLOW_QUEUE_DEPTH)
        response = await self._post()
        self.assertEqual(200, response.status)

    async def test_a_request_that_already_crossed_is_never_sent_back(self):
        self.router._queue_waiting[self.MODEL] = 50
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post({mm.OVERFLOW_HEADER: "1"})
        self.assertEqual(200, response.status)

    async def test_sticky_hit_stays_local_to_keep_its_prefix_cache(self):
        self.router._queue_waiting[self.MODEL] = 50
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4), \
             patch.object(mm, "_sticky_slot_for", lambda body, headers: (1, "task")):
            response = await self._post()
        self.assertEqual(200, response.status)
        self.assertEqual([1], self.proxied)


class OverflowQueueSampleTests(unittest.IsolatedAsyncioTestCase):
    """The saturation sampler is what keeps _queue_waiting current."""

    async def test_sampler_records_the_summed_queue_and_forgets_stopped_models(self):
        model = "qwen3.8-27b"
        config = mm.ModelConfig(script="unused.sh", served_name=model, max_num_seqs=8)
        router = mm.DynamicRouter([mm.GpuSlot(0, 0, 9000)], {model: config})

        class Backend:
            async def queue_depth(self):
                return 6

        running = [[Backend(), Backend()], []]
        router._running_backends = lambda _m: running[0]
        sleeps = 0

        async def fake_sleep(_s):
            nonlocal sleeps
            sleeps += 1
            if sleeps == 2:
                self.assertEqual(12, router._queue_waiting[model])
                running[0] = []
            if sleeps == 3:
                self.assertNotIn(model, router._queue_waiting)
                raise mm.asyncio.CancelledError

        with patch.object(mm.asyncio, "sleep", fake_sleep):
            await router._saturation_loop()
        self.assertEqual(3, sleeps)


if __name__ == "__main__":
    unittest.main()
