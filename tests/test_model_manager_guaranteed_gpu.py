"""GUARANTEED_GPU_UUIDS: cards where a primary spawn first stops every process
that is not one of this gateway's backends."""

import signal
import unittest
from unittest.mock import patch

import model_manager as mm
from model_manager import DynamicRouter, GpuBackend, GpuSlot

CHAT = "qwen3.8-27b"
UUIDS = "index, uuid\n0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n"


class ParseTests(unittest.TestCase):
    def test_uuids_resolve_to_nvidia_smi_indices(self):
        table = mm._parse_gpu_uuid_table(UUIDS)
        self.assertEqual(frozenset({1, 2}), mm._resolve_guaranteed_gpus(" GPU-bbb,GPU-ccc ", table))
        self.assertEqual(frozenset(), mm._resolve_guaranteed_gpus("", {}))

    def test_unknown_uuid_stops_startup(self):
        with self.assertRaises(ValueError):
            mm._resolve_guaranteed_gpus("GPU-zzz", mm._parse_gpu_uuid_table(UUIDS))


class FakeGpu:
    """Processes on one card: pid -> (pgid, signal that ends it)."""

    def __init__(self, procs):
        self.procs = dict(procs)
        self.sent = []

    def pids(self, gpu_id):
        return list(self.procs)

    def pgid(self, pid):
        return self.procs[pid][0] if pid in self.procs else None

    def send(self, pid, sig):
        self.sent.append((pid, sig))
        ends_on = self.procs[pid][1]
        if ends_on is not None and sig in (ends_on, signal.SIGKILL):
            del self.procs[pid]


class ClearGpuTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        mm._gpu_clear_failed_at.clear()

    def _backend(self, gpu_id=1, live=(500,), coresident=False):
        backend = GpuBackend(CHAT, "run.sh", CHAT, GpuSlot(0, gpu_id, 9000), coresident=coresident)
        backend.router = type("R", (), {"live_pgids": lambda _self: set(live)})()
        return backend

    async def _clear(self, gpu, backend, guaranteed=frozenset({1})):
        with patch.object(mm, "GUARANTEED_GPU_IDS", guaranteed), \
                patch.object(mm, "GPU_CLEAR_TERM_SECONDS", 0.0), \
                patch.object(mm, "GPU_CLEAR_KILL_SECONDS", 0.0), \
                patch.object(mm, "_gpu_compute_pids", gpu.pids), \
                patch.object(mm, "_signal_pid", gpu.send), \
                patch.object(mm, "_describe_pid", return_value="someone: python"), \
                patch.object(GpuBackend, "_pgid_of", staticmethod(gpu.pgid)):
            await backend._clear_guaranteed_gpu()

    async def test_stops_foreign_processes_and_spares_gateway_backends(self):
        gpu = FakeGpu({100: (100, signal.SIGTERM), 501: (500, signal.SIGTERM)})
        await self._clear(gpu, self._backend())
        self.assertEqual([(100, signal.SIGTERM)], gpu.sent)

    async def test_escalates_to_sigkill(self):
        gpu = FakeGpu({100: (100, signal.SIGKILL)})
        await self._clear(gpu, self._backend())
        self.assertEqual([(100, signal.SIGTERM), (100, signal.SIGKILL)], gpu.sent)

    async def test_a_process_that_survives_fails_the_spawn(self):
        gpu = FakeGpu({100: (100, None)})
        with self.assertRaises(RuntimeError):
            await self._clear(gpu, self._backend())
        self.assertIn(1, mm._gpu_clear_failed_at)

    async def test_other_cards_and_coresident_spawns_are_untouched(self):
        for backend in (self._backend(gpu_id=0), self._backend(coresident=True)):
            gpu = FakeGpu({100: (100, signal.SIGTERM)})
            await self._clear(gpu, backend)
            self.assertEqual([], gpu.sent)

    async def test_without_a_router_nothing_is_killed(self):
        backend = self._backend()
        backend.router = None
        gpu = FakeGpu({100: (100, signal.SIGTERM)})
        await self._clear(gpu, backend)
        self.assertEqual([], gpu.sent)

    async def test_spawn_clears_before_checking_for_leftovers(self):
        order = []
        backend = self._backend()
        async def clear():
            order.append("clear")

        async def reclaim(_backend):
            order.append("reclaim")

        backend.router.reclaim_coresident_for = reclaim

        def check():
            order.append("check")
            raise RuntimeError("stop here")

        with patch.object(mm, "GUARANTEED_GPU_IDS", frozenset({1})), \
                patch.object(backend, "_ensure_session"), patch.object(backend, "_ensure_idle_task"), \
                patch.object(backend, "_clear_guaranteed_gpu", clear), \
                patch.object(backend, "_check_gpu_free", check):
            with self.assertRaises(RuntimeError):
                await backend._spawn_locked()
        self.assertEqual(["clear", "reclaim", "check"], order)
        self.assertIn(1, mm._gpu_clear_failed_at)

    async def test_an_unreadable_card_is_marked(self):
        def broken(_gpu_id):
            raise OSError("nvidia-smi")

        with patch.object(mm, "GUARANTEED_GPU_IDS", frozenset({1})), \
                patch.object(mm, "_gpu_compute_pids", broken):
            await self._backend()._clear_guaranteed_gpu()
        self.assertIn(1, mm._gpu_clear_failed_at)


class SlotChoiceTests(unittest.TestCase):
    def setUp(self):
        mm._gpu_clear_failed_at.clear()
        self.slots = [GpuSlot(0, 0, 9000), GpuSlot(1, 1, 9010)]

    def _first(self):
        # The guaranteed card (1) is the fuller one.
        with patch.object(mm, "GUARANTEED_GPU_IDS", frozenset({1})), \
                patch.object(mm, "_gpu_free_mib", side_effect=lambda g: {0: 30000.0, 1: 100.0}[g]):
            return DynamicRouter._by_free_vram(self.slots)[0].gpu_id

    def test_guaranteed_card_is_chosen_even_when_occupied(self):
        self.assertEqual(1, self._first())

    def test_an_unreadable_guaranteed_card_sorts_last(self):
        with patch.object(mm, "GUARANTEED_GPU_IDS", frozenset({1})), \
                patch.object(mm, "_gpu_free_mib", side_effect=lambda g: {0: 100.0, 1: None}[g]):
            self.assertEqual(0, DynamicRouter._by_free_vram(self.slots)[0].gpu_id)

    def test_preference_stays_lost_past_the_scale_out_cooldown(self):
        self.assertGreater(mm.GPU_CLEAR_RETRY_SECONDS, DynamicRouter.SCALE_OUT_COOLDOWN)

    def test_a_card_that_could_not_be_cleared_loses_its_preference(self):
        mm._gpu_clear_failed_at[1] = mm.time.monotonic()
        self.assertEqual(0, self._first())


class SignalTests(unittest.TestCase):
    def test_another_users_process_is_signalled_through_sudo(self):
        with patch.object(mm.os, "kill", side_effect=PermissionError), \
                patch.object(mm.subprocess, "run") as run:
            mm._signal_pid(1234, signal.SIGTERM)
        self.assertEqual(["sudo", "-n", "kill", "-15", "1234"], run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
