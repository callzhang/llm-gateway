"""Scale out when every replica is busy, even with no queue.

On 2026-11-07 qwen3.8-27b served 4-5 long-context agent requests on one slot for
half an hour: vLLM admitted them all (waiting ~0), each ran slowly, and requests hit
the 600s deadline while the other GPU was empty. The queue tiers never fired.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import model_manager as mm

MODEL = "qwen3.8-27b"
OTHER = "qwen3.6-35b"


def _router(n_slots):
    configs = {
        MODEL: mm.ModelConfig(script="unused.sh", served_name=MODEL, max_num_seqs=8),
        OTHER: mm.ModelConfig(script="unused.sh", served_name=OTHER, max_num_seqs=8),
    }
    slots = [mm.GpuSlot(i, i, 9300 + 10 * i) for i in range(n_slots)]
    return mm.DynamicRouter(slots, configs)


def _backend(router, slot, model=MODEL, *, active=0):
    b = router._make_backend(model, slot)
    b._ready = True
    b.process = SimpleNamespace(pid=4242 + slot.slot_id, poll=lambda: None)
    b._active_requests = active
    b.queue_depth = AsyncMock(return_value=0)
    slot.backend = b
    return b


async def _run_sampler(router, ticks_max=40):
    triggered = []
    router._trigger_scale_out = lambda name, free_only=False: triggered.append((name, free_only))
    ticks = 0
    clock = [1000.0]

    async def fake_sleep(_s):
        nonlocal ticks
        ticks += 1
        clock[0] += 10.0
        if triggered or ticks > ticks_max:
            raise asyncio.CancelledError

    with patch.object(mm.asyncio, "sleep", fake_sleep), \
         patch.object(mm.time, "monotonic", lambda: clock[0]):
        await router._saturation_loop()
    return triggered, ticks


class BusySamplerTests(unittest.IsolatedAsyncioTestCase):
    async def test_busy_replica_with_empty_queue_triggers_free_slot_scale_out(self):
        router = _router(2)
        _backend(router, router.slots[0], active=mm.SCALE_OUT_BUSY_REQUESTS)
        triggered, ticks = await _run_sampler(router)
        self.assertEqual([(MODEL, True)], triggered)
        # About SCALE_OUT_BUSY_SUSTAIN seconds of sampling, not the 300s queue tier.
        self.assertLessEqual(ticks * 10, mm.SCALE_OUT_BUSY_SUSTAIN + 20)

    async def test_below_the_busy_threshold_nothing_accrues(self):
        router = _router(2)
        _backend(router, router.slots[0], active=mm.SCALE_OUT_BUSY_REQUESTS - 1)
        triggered, _ = await _run_sampler(router)
        self.assertEqual([], triggered)
        self.assertNotIn(MODEL, router._scale_accrual)

    async def test_one_idle_replica_means_not_busy(self):
        router = _router(3)
        _backend(router, router.slots[0], active=mm.SCALE_OUT_BUSY_REQUESTS + 2)
        _backend(router, router.slots[1], active=0)
        triggered, _ = await _run_sampler(router)
        self.assertEqual([], triggered)

    async def test_queue_backlog_still_allows_eviction(self):
        router = _router(2)
        b = _backend(router, router.slots[0], active=mm.SCALE_OUT_BUSY_REQUESTS)
        b.queue_depth = AsyncMock(return_value=4)
        triggered, _ = await _run_sampler(router)
        self.assertEqual([(MODEL, False)], triggered)

    async def test_disabled_with_zero(self):
        router = _router(2)
        _backend(router, router.slots[0], active=10)
        with patch.object(mm, "SCALE_OUT_BUSY_REQUESTS", 0):
            triggered, _ = await _run_sampler(router)
        self.assertEqual([], triggered)


class BusyScaleOutNeverEvictsTests(unittest.IsolatedAsyncioTestCase):
    async def _scale_out(self, router, free_only):
        spawned = []

        async def ensure_running(self_b):
            spawned.append(self_b.slot.slot_id)
            self_b._ready = True

        with patch.object(mm, "_gpu_free_mib", return_value=40000.0), \
             patch.object(mm, "_gpu_total_mib", return_value=40960.0), \
             patch.object(mm, "_gpu_vllm_used_mib", return_value=0.0), \
             patch.object(mm.GpuBackend, "start", AsyncMock()), \
             patch.object(mm.GpuBackend, "_ensure_running", ensure_running), \
             patch.object(mm.GpuBackend, "_kill_process_locked", AsyncMock()):
            await router._maybe_scale_out(MODEL, free_only=free_only)
        return spawned

    async def test_free_slot_is_taken(self):
        router = _router(2)
        _backend(router, router.slots[0], active=5)
        self.assertEqual([1], await self._scale_out(router, free_only=True))

    async def test_idle_foreign_model_is_not_evicted(self):
        router = _router(2)
        _backend(router, router.slots[0], active=5)
        other = _backend(router, router.slots[1], OTHER)
        self.assertEqual([], await self._scale_out(router, free_only=True))
        self.assertIs(other, router.slots[1].backend)

    async def test_queue_scale_out_still_evicts_idle_foreign_model(self):
        router = _router(2)
        _backend(router, router.slots[0], active=5)
        _backend(router, router.slots[1], OTHER)
        self.assertEqual([1], await self._scale_out(router, free_only=False))


if __name__ == "__main__":
    unittest.main()
