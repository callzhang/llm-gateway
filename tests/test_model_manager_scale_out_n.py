"""Scale-out past two replicas.

On a 2-GPU host "already on 2 slots" and "no free slot" were the same thing,
so scale-out stopped at the second replica.  The A100 region has seven GPUs:
a model under sustained backlog keeps adding replicas while free slots with
room exist, one spawn at a time.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import model_manager as mm

MODEL = "qwen3.8-27b"


def _router(n_slots):
    config = mm.ModelConfig(script="unused.sh", served_name=MODEL, max_num_seqs=8)
    slots = [mm.GpuSlot(i, i, 9300 + 10 * i) for i in range(n_slots)]
    return mm.DynamicRouter(slots, {MODEL: config})


def _backend(router, slot, *, ready=True):
    b = router._make_backend(MODEL, slot)
    b._ready = ready
    b.process = SimpleNamespace(pid=4242 + slot.slot_id, poll=lambda: None)
    slot.backend = b
    return b


def _plenty_of_vram():
    return (
        patch.object(mm, "_gpu_free_mib", return_value=40000.0),
        patch.object(mm, "_gpu_total_mib", return_value=40960.0),
        patch.object(mm, "_gpu_vllm_used_mib", return_value=0.0),
    )


class ScaleOutBeyondTwoTests(unittest.IsolatedAsyncioTestCase):
    async def _scale_out(self, router):
        spawned = []

        async def ensure_running(self_b):
            spawned.append(self_b.slot.slot_id)
            self_b.process = SimpleNamespace(pid=5000, poll=lambda: None)
            self_b._ready = True

        a, b, c = _plenty_of_vram()
        with a, b, c, \
             patch.object(mm.GpuBackend, "start", AsyncMock()), \
             patch.object(mm.GpuBackend, "_ensure_running", ensure_running):
            await router._maybe_scale_out(MODEL)
        return spawned

    async def test_two_running_replicas_grow_to_a_third(self):
        router = _router(4)
        _backend(router, router.slots[0])
        _backend(router, router.slots[1])
        spawned = await self._scale_out(router)
        self.assertEqual(1, len(spawned))
        self.assertIn(spawned[0], (2, 3))
        self.assertEqual(3, len(router._running_backends(MODEL)))

    async def test_no_second_spawn_while_one_is_still_starting(self):
        router = _router(4)
        _backend(router, router.slots[0])
        _backend(router, router.slots[1], ready=False)   # mid-spawn replica
        spawned = await self._scale_out(router)
        self.assertEqual([], spawned)

    async def test_stops_when_every_slot_holds_the_model(self):
        router = _router(3)
        for s in router.slots:
            _backend(router, s)
        spawned = await self._scale_out(router)
        self.assertEqual([], spawned)


class SamplerAccruesWithManyReplicasTests(unittest.IsolatedAsyncioTestCase):
    async def test_backlog_on_two_replicas_still_triggers_scale_out(self):
        router = _router(4)
        r0 = _backend(router, router.slots[0])
        r1 = _backend(router, router.slots[1])
        for r in (r0, r1):
            r.queue_depth = AsyncMock(return_value=4)    # 8 waiting in total
        triggered = []
        router._trigger_scale_out = triggered.append
        ticks = 0
        clock = [1000.0]

        async def fake_sleep(_s):
            nonlocal ticks
            ticks += 1
            clock[0] += 10.0
            if triggered or ticks > 30:
                raise asyncio.CancelledError

        with patch.object(mm.asyncio, "sleep", fake_sleep), \
             patch.object(mm.time, "monotonic", lambda: clock[0]):
            await router._saturation_loop()
        self.assertEqual([MODEL], triggered)

    async def test_no_accrual_while_a_replica_is_starting(self):
        router = _router(4)
        r0 = _backend(router, router.slots[0])
        _backend(router, router.slots[1], ready=False)
        r0.queue_depth = AsyncMock(return_value=8)
        triggered = []
        router._trigger_scale_out = triggered.append
        ticks = 0
        clock = [1000.0]

        async def fake_sleep(_s):
            nonlocal ticks
            ticks += 1
            clock[0] += 10.0
            if ticks > 30:
                raise asyncio.CancelledError

        with patch.object(mm.asyncio, "sleep", fake_sleep), \
             patch.object(mm.time, "monotonic", lambda: clock[0]):
            await router._saturation_loop()
        self.assertEqual([], triggered)
        self.assertNotIn(MODEL, router._scale_accrual)


if __name__ == "__main__":
    unittest.main()
