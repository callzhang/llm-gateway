"""Cross-region overflow: a long local queue turns new work away with a 503.

Each region runs its own gateway stack.  LiteLLM lists the other region's
gateway as an order-2 deployment of the same model, so a 503 from this
model_manager is what moves a request across the border.  The rule here:

  * a sticky hit keeps its prefix cache local, unless its own slot already has
    requests queued behind the gate — then it is turned away like any other, since
    waiting there costs more than recomputing the history in the other region;
  * a sticky binding lapses after STICKY_IDLE_TTL without a request, because the
    prefix cache it protects has long been evicted by then;
  * a request that already crossed (x-gateway-overflow header, set by the
    peer's LiteLLM deployment, or simply coming from another host) is never
    turned away again, so the two regions cannot bounce a request between them;
  * OVERFLOW_QUEUE_DEPTH is per running instance: the model's queue, summed over
    its instances, is compared with depth x instances, so scaling out does not
    make overflow easier (2026-11-22: four A100 replicas each queueing one
    request already sent ~40 requests a minute across with depth 4);
  * OVERFLOW_QUEUE_DEPTH=0 (the default) disables the rule entirely.
"""

import json
import unittest
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import model_manager as mm


class FakeBackend:
    """A running backend with requests queued at its gate, as {priority: count}."""

    def __init__(self, slot, proxy):
        self.slot, self.proxy, self.queued = slot, proxy, {}

    @property
    def local_waiting(self) -> int:
        return sum(self.queued.values())

    @local_waiting.setter
    def local_waiting(self, n: int) -> None:
        self.queued = {mm.DEFAULT_REQUEST_PRIORITY: n}

    def waiting_behind(self, priority: int) -> int:
        return sum(n for p, n in self.queued.items() if p > priority)


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
            return FakeBackend(slot, proxy)

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

    async def _post(self, headers=None, **extra):
        payload = {"model": self.MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}
        return await self.client.post(
            "/v1/chat/completions", json=payload, headers=headers or {},
        )

    async def test_a_request_past_the_gate_limit_gets_queue_overflow(self):
        async def overflowing(request, body, priority=mm.DEFAULT_REQUEST_PRIORITY):
            raise mm.QueueOverflow("waited 90s for a place on slot 0; send it to the other region.")

        self.backends[0] = type("B", (), {"slot": self.router.slots[0], "proxy": staticmethod(overflowing)})()
        response = await self._post()
        self.assertEqual(503, response.status)
        self.assertEqual("queue_overflow", (await response.json())["error"]["type"])

    async def test_long_queue_turns_new_work_away_with_a_503(self):
        self.router._queue_waiting[self.MODEL] = 8  # depth 4 x 2 instances
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(503, response.status)
        self.assertEqual("queue_overflow", (await response.json())["error"]["type"])
        self.assertEqual([], self.proxied)

    async def test_queue_below_the_depth_is_served_locally(self):
        self.router._queue_waiting[self.MODEL] = 7  # under depth 4 x 2 instances
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(200, response.status)
        self.assertEqual([0], self.proxied)

    async def test_only_requests_queued_ahead_count_toward_overflow(self):
        """A request is not sent across for a queue it would jump: the gate admits it
        before every waiter with a larger priority number."""
        self.router._queue_waiting[self.MODEL] = 8  # depth 4 x 2 instances
        self.backends[0].queued = {100: 3, 200: 2}
        self.backends[1].queued = {100: 1, 200: 2}
        cases = (
            (0, 200, [0]),      # granted: all 8 queued are behind it
            (100, 200, [0]),    # normal: 4 at 200 behind it, 4 at 100 ahead (< 8) -> local
            (200, 503, []),     # eval: all 8 are ahead of it (equal priority is FIFO)
        )
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            for priority, status, proxied in cases:
                with self.subTest(priority=priority):
                    self.proxied.clear()
                    response = await self._post(priority=priority)
                    self.assertEqual(status, response.status)
                    self.assertEqual(proxied, self.proxied)

    async def test_a_request_without_priority_counts_waiters_at_200_as_behind(self):
        self.router._queue_waiting[self.MODEL] = 8
        self.backends[0].queued = {200: 8}
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(200, response.status)

    def _run_on(self, instances):
        slots = [mm.GpuSlot(i, i, 9000 + 10 * i) for i in range(instances)]
        backends = [
            FakeBackend(s, self.backends[0].proxy)
            for s in slots
        ]

        async def get_or_start(_model):
            return backends

        self.router._get_or_start = get_or_start

    async def test_four_instances_each_queueing_one_request_are_served_locally(self):
        self._run_on(4)
        self.router._queue_waiting[self.MODEL] = 4
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(200, response.status)
        self.assertEqual([0], self.proxied)

    async def test_four_instances_overflow_at_depth_times_four(self):
        self._run_on(4)
        self.router._queue_waiting[self.MODEL] = 16
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(503, response.status)
        self.assertIn("16 requests queued on 4 instances", (await response.json())["error"]["message"])
        self.assertEqual([], self.proxied)

    async def test_one_instance_overflows_at_the_depth(self):
        self._run_on(1)
        self.router._queue_waiting[self.MODEL] = 4
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            response = await self._post()
        self.assertEqual(503, response.status)
        self.assertEqual([], self.proxied)

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

    async def test_a_request_from_another_host_is_never_sent_back(self):
        self.router._queue_waiting[self.MODEL] = 50
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4), \
             patch.object(web.BaseRequest, "remote", "100.118.88.28"):
            response = await self._post()
        self.assertEqual(200, response.status)

    async def _post_sticky_to_slot_1(self, headers=None):
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4), \
             patch.object(mm, "_sticky_slot_for", lambda body, headers: (1, "task")):
            return await self._post(headers)

    async def test_sticky_hit_on_a_free_slot_stays_local_to_keep_its_prefix_cache(self):
        self.router._queue_waiting[self.MODEL] = 50
        response = await self._post_sticky_to_slot_1()
        self.assertEqual(200, response.status)
        self.assertEqual([1], self.proxied)

    async def test_sticky_hit_whose_slot_is_queueing_is_turned_away(self):
        self.router._queue_waiting[self.MODEL] = 50
        self.backends[1].local_waiting = 3
        response = await self._post_sticky_to_slot_1()
        self.assertEqual(503, response.status)
        self.assertEqual("queue_overflow", (await response.json())["error"]["type"])
        self.assertEqual([], self.proxied)

    async def test_sticky_hit_stays_when_the_model_queue_is_below_the_depth(self):
        self.router._queue_waiting[self.MODEL] = 3
        self.backends[1].local_waiting = 3
        response = await self._post_sticky_to_slot_1()
        self.assertEqual(200, response.status)
        self.assertEqual([1], self.proxied)

    async def test_sticky_hit_stays_when_its_slot_queues_only_behind_it(self):
        self.router._queue_waiting[self.MODEL] = 8
        self.backends[1].queued = {200: 3}
        with patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4), \
             patch.object(mm, "_sticky_slot_for", lambda body, headers: (1, "task")):
            response = await self._post(priority=0)
        self.assertEqual(200, response.status)
        self.assertEqual([1], self.proxied)

    async def test_a_crossed_request_is_served_even_on_a_queueing_sticky_slot(self):
        self.router._queue_waiting[self.MODEL] = 50
        self.backends[1].local_waiting = 3
        response = await self._post_sticky_to_slot_1({mm.OVERFLOW_HEADER: "1"})
        self.assertEqual(200, response.status)
        self.assertEqual([1], self.proxied)


class StickyAffinityExpiryTests(unittest.TestCase):
    """A binding is only worth keeping while the prefix cache it protects may survive."""

    def setUp(self):
        mm._task_affinity.clear()
        self.addCleanup(mm._task_affinity.clear)
        self.headers = {"x-task-id": "task-1"}

    def _lookup(self, now):
        with patch.object(mm.time, "monotonic", return_value=now):
            return mm._sticky_slot_for(None, self.headers)[0]

    def _bind(self, now, slot_id):
        with patch.object(mm.time, "monotonic", return_value=now):
            mm._set_task_affinity("task-1", slot_id)

    def test_binding_is_kept_within_the_idle_ttl(self):
        self._bind(1000.0, 1)
        self.assertEqual(1, self._lookup(1000.0 + mm.STICKY_IDLE_TTL))

    def test_binding_lapses_after_the_idle_ttl_and_is_dropped(self):
        self._bind(1000.0, 1)
        self.assertIsNone(self._lookup(1000.0 + mm.STICKY_IDLE_TTL + 1))
        self.assertNotIn("task-1", mm._task_affinity)

    def test_each_request_restarts_the_idle_clock(self):
        self._bind(1000.0, 1)
        self._bind(1000.0 + mm.STICKY_IDLE_TTL - 1, 1)
        self.assertEqual(1, self._lookup(1000.0 + 2 * mm.STICKY_IDLE_TTL - 2))


class AdminActionsAreLoopbackOnlyTests(unittest.IsolatedAsyncioTestCase):
    """The peer region reaches :8002 over Tailscale for overflow, and the peer
    host is shared — its users must not be able to kill or switch our slots."""

    async def asyncSetUp(self):
        config = mm.ModelConfig(script="unused.sh", served_name="m", max_num_seqs=8)
        self.router = mm.DynamicRouter([mm.GpuSlot(0, 0, 9000)], {"m": config})
        self.killed: list[int] = []

        async def admin_kill(slot_id):
            self.killed.append(slot_id)

        self.router.admin_kill = admin_kill
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", self.router.handle)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_remote_client_gets_403(self):
        with patch.object(web.BaseRequest, "remote", "100.118.88.28"):
            response = await self.client.post("/admin/kill", json={"slot_id": 0})
        self.assertEqual(403, response.status)
        await mm.asyncio.sleep(0)
        self.assertEqual([], self.killed)

    async def test_loopback_client_is_allowed(self):
        response = await self.client.post("/admin/kill", json={"slot_id": 0})
        self.assertEqual(200, response.status)
        await mm.asyncio.sleep(0)
        self.assertEqual([0], self.killed)


class OverflowQueueSampleTests(unittest.IsolatedAsyncioTestCase):
    """The saturation sampler is what keeps _queue_waiting current."""

    async def test_sampler_records_the_summed_queue_and_forgets_stopped_models(self):
        model = "qwen3.8-27b"
        config = mm.ModelConfig(script="unused.sh", served_name=model, max_num_seqs=8)
        router = mm.DynamicRouter([mm.GpuSlot(0, 0, 9000)], {model: config})

        class Backend:
            _active_requests = 0

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
