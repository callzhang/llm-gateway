from __future__ import annotations

import asyncio
import json
import unittest
from contextlib import suppress

import aiohttp
from aiohttp import web

import model_manager as mm


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


class PriorityGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_below_capacity_admits_without_waiting(self):
        gate = mm.PriorityGate(2)
        await asyncio.wait_for(gate.acquire(100), 1)
        await asyncio.wait_for(gate.acquire(200), 1)
        self.assertEqual(2, gate.in_flight)
        self.assertEqual(0, gate.waiting)

    async def test_full_gate_admits_lowest_priority_value_first_then_arrival_order(self):
        gate = mm.PriorityGate(1)
        await gate.acquire(0)
        order: list[str] = []

        async def waiter(name: str, priority: int):
            await gate.acquire(priority)
            order.append(name)

        tasks = [
            asyncio.create_task(waiter("eval-1", 200)),
            asyncio.create_task(waiter("debug", 100)),
            asyncio.create_task(waiter("eval-2", 200)),
            asyncio.create_task(waiter("live", 0)),
        ]
        await _settle()
        self.assertEqual(4, gate.waiting)
        for _ in tasks:
            gate.release()
            await _settle()
        await asyncio.gather(*tasks)
        self.assertEqual(["live", "debug", "eval-1", "eval-2"], order)

    async def test_low_priority_is_not_promoted_by_waiting(self):
        gate = mm.PriorityGate(1)
        await gate.acquire(0)
        order: list[str] = []

        async def waiter(name: str, priority: int):
            await gate.acquire(priority)
            order.append(name)

        old_eval = asyncio.create_task(waiter("old-eval", 200))
        await _settle()
        await asyncio.sleep(0.05)
        new_live = asyncio.create_task(waiter("new-live", 0))
        await _settle()
        gate.release()
        await _settle()
        self.assertEqual(["new-live"], order)
        gate.release()
        await asyncio.gather(old_eval, new_live)
        self.assertEqual(["new-live", "old-eval"], order)

    async def test_cancelled_waiter_leaves_the_queue_and_does_not_take_a_slot(self):
        gate = mm.PriorityGate(1)
        await gate.acquire(0)
        cancelled = asyncio.create_task(gate.acquire(0))
        later = asyncio.create_task(gate.acquire(200))
        await _settle()
        cancelled.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await cancelled
        self.assertEqual(1, gate.waiting)
        gate.release()
        await asyncio.wait_for(later, 1)
        self.assertEqual(1, gate.in_flight)

    async def test_grant_racing_a_cancellation_releases_the_slot(self):
        gate = mm.PriorityGate(1)
        await gate.acquire(0)
        waiter = asyncio.create_task(gate.acquire(0))
        await _settle()
        gate.release()          # grants the slot to the waiter's future
        waiter.cancel()         # cancelled before it resumes
        with suppress(asyncio.CancelledError):
            await waiter
        self.assertEqual(0, gate.in_flight)
        await asyncio.wait_for(gate.acquire(0), 1)

    async def test_waiting_by_priority_is_reported(self):
        gate = mm.PriorityGate(1)
        await gate.acquire(0)
        tasks = [asyncio.create_task(gate.acquire(p)) for p in (200, 200, 0)]
        await _settle()
        self.assertEqual({0: 1, 200: 2}, gate.waiting_by_priority())
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class RequestPriorityTests(unittest.TestCase):
    def test_missing_priority_is_the_default(self):
        self.assertEqual(100, mm.request_priority({"model": "m"}))
        self.assertEqual(100, mm.request_priority(None))

    def test_integer_priority_in_range_is_used(self):
        self.assertEqual(0, mm.request_priority({"priority": 0}))
        self.assertEqual(255, mm.request_priority({"priority": 255}))

    def test_invalid_priority_is_rejected(self):
        for bad in (-1, 256, 1.5, "high", True, None):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    mm.request_priority({"priority": bad})


class BackendGateIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def _upstream(self, release: asyncio.Event, seen: list[dict]):
        async def handler(request: web.Request):
            seen.append(json.loads(await request.read()))
            await release.wait()
            return web.json_response({"ok": True})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        return runner, port

    async def test_backend_admits_at_most_max_num_seqs_and_orders_the_rest(self):
        release, seen = asyncio.Event(), []
        runner, port = await self._upstream(release, seen)
        try:
            backend = mm.GpuBackend(
                "qwen3.8-27b", "unused.sh", "qwen3.8-27b", mm.GpuSlot(0, 0, port), max_num_seqs=1
            )
            backend.slot.backend = backend
            backend._ready = True
            backend.process = type("P", (), {"poll": lambda self: None})()

            async def forward(request, body):
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://127.0.0.1:{port}/v1/chat/completions", data=body
                    ) as upstream:
                        await upstream.read()
                return web.Response(status=200)

            backend._forward = forward

            async def call(priority: int):
                body = json.dumps({"model": "qwen3.8-27b", "messages": [], "tag": priority}).encode()
                request = type("R", (), {
                    "path_qs": "/v1/chat/completions", "method": "POST", "headers": {},
                    "transport": type("T", (), {"is_closing": lambda self: False})(),
                })()
                return await backend.proxy(request, body, priority)

            first = asyncio.create_task(call(200))
            await asyncio.sleep(0.1)
            low = asyncio.create_task(call(200))
            high = asyncio.create_task(call(0))
            await asyncio.sleep(0.1)
            self.assertEqual(1, len(seen), "the gate must keep vLLM's in-flight at max_num_seqs")
            self.assertEqual(2, backend._gate.waiting)
            release.set()
            await asyncio.wait_for(asyncio.gather(first, low, high, return_exceptions=True), 5)
            self.assertEqual([200, 0, 200], [b["tag"] for b in seen])
        finally:
            release.set()
            await runner.cleanup()


if __name__ == "__main__":
    unittest.main()
