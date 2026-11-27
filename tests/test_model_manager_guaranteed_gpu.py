"""GUARANTEED_GPU_UUIDS: cards where a primary spawn first stops every process
that is not one of this gateway's backends."""

import asyncio
import signal
import subprocess
import sys
import time
import unittest
from unittest.mock import patch

import model_manager as mm
from model_manager import DynamicRouter, GpuBackend, GpuSlot

CHAT = "qwen3.8-27b"
UUIDS = "index, uuid\n0, GPU-aaa\n1, GPU-bbb\n2, GPU-ccc\n"


def cards(order=("GPU-aaa", "GPU-bbb", "GPU-ccc"), guaranteed=("GPU-bbb",)):
    """Patch which card each nvidia-smi index is, and which are guaranteed."""
    return patch.multiple(mm, GUARANTEED_GPU_UUIDS=frozenset(guaranteed),
                          _gpu_uuid=lambda i: order[i] if i < len(order) else None)


class ParseTests(unittest.TestCase):
    def test_uuids_are_checked_against_the_host(self):
        table = mm._parse_gpu_uuid_table(UUIDS)
        self.assertEqual(frozenset({"GPU-bbb", "GPU-ccc"}), mm._parse_guaranteed_uuids(" GPU-bbb,GPU-ccc ", table))

    def test_unknown_uuid_stops_startup(self):
        with self.assertRaises(ValueError):
            mm._parse_guaranteed_uuids("GPU-zzz", mm._parse_gpu_uuid_table(UUIDS))


class FakeGpu:
    """Processes on one card: pid -> (pgid, signal that ends it)."""

    def __init__(self, procs):
        self.procs = dict(procs)
        self.sent = []

    def pids(self, gpu_id):
        return list(self.procs)

    def pgid(self, pid):
        return self.procs[pid][0] if pid in self.procs else None

    def send(self, pid, sig, started=None):
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

    async def _clear(self, gpu, backend, **host):
        with cards(**host), \
                patch.object(mm, "GPU_CLEAR_TERM_SECONDS", 0.0), \
                patch.object(mm, "GPU_CLEAR_KILL_SECONDS", 0.0), \
                patch.object(mm, "_gpu_compute_pids", gpu.pids), \
                patch.object(mm, "_signal_pid", gpu.send), \
                patch.object(mm, "_pid_started", lambda pid: "1"), \
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
        self.assertIn("GPU-bbb", mm._gpu_clear_failed_at)

    async def test_a_card_that_took_over_the_index_is_left_alone(self):
        gpu = FakeGpu({100: (100, signal.SIGTERM)})
        await self._clear(gpu, self._backend(), order=("GPU-new", "GPU-aaa", "GPU-bbb"))
        self.assertEqual([], gpu.sent)

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

        with cards(), \
                patch.object(backend, "_ensure_session"), patch.object(backend, "_ensure_idle_task"), \
                patch.object(backend, "_clear_guaranteed_gpu", clear), \
                patch.object(backend, "_check_gpu_free", check):
            with self.assertRaises(RuntimeError):
                await backend._spawn_locked()
        self.assertEqual(["clear", "reclaim", "check"], order)
        self.assertIn("GPU-bbb", mm._gpu_clear_failed_at)

    async def test_an_unreadable_card_fails_the_spawn_and_is_marked(self):
        def broken(_gpu_id):
            raise OSError("nvidia-smi")

        with cards(), patch.object(mm, "_gpu_compute_pids", broken), self.assertRaises(RuntimeError):
            await self._backend()._clear_guaranteed_gpu()
        self.assertIn("GPU-bbb", mm._gpu_clear_failed_at)

    async def test_a_backend_that_started_during_the_card_query_is_spared(self):
        live = {500}
        backend = self._backend()
        backend.router = type("R", (), {"live_pgids": lambda _self: set(live)})()
        gpu = FakeGpu({600: (600, signal.SIGTERM)})

        def pids(gpu_id):
            live.add(600)   # an ASR engine came up while nvidia-smi was running
            return gpu.pids(gpu_id)

        with cards(), patch.object(mm, "_gpu_compute_pids", pids), patch.object(mm, "_signal_pid", gpu.send), \
                patch.object(GpuBackend, "_pgid_of", staticmethod(gpu.pgid)):
            await backend._clear_guaranteed_gpu()
        self.assertEqual([], gpu.sent)

    async def test_a_card_that_cannot_be_identified_fails_the_spawn(self):
        gpu = FakeGpu({100: (100, signal.SIGTERM)})
        with patch.multiple(mm, GUARANTEED_GPU_UUIDS=frozenset({"GPU-bbb"}), _gpu_uuid=lambda i: None), \
                patch.object(mm, "_signal_pid", gpu.send), self.assertRaises(RuntimeError):
            await self._backend()._clear_guaranteed_gpu()
        self.assertEqual([], gpu.sent)

    async def test_a_slow_card_query_does_not_stall_other_requests(self):
        ticks = []

        async def ticker():
            while True:
                ticks.append(1)
                await asyncio.sleep(0.01)

        def slow(_gpu_id):
            time.sleep(0.3)
            return []

        task = asyncio.create_task(ticker())
        await asyncio.sleep(0)
        with cards(), patch.object(mm, "_gpu_compute_pids", slow):
            await self._backend()._clear_guaranteed_gpu()
        task.cancel()
        self.assertGreater(len(ticks), 10)


class SlotChoiceTests(unittest.TestCase):
    def setUp(self):
        mm._gpu_clear_failed_at.clear()
        self.slots = [GpuSlot(0, 0, 9000), GpuSlot(1, 1, 9010)]

    def _first(self):
        # The guaranteed card (1) is the fuller one.
        with cards(), \
                patch.object(mm, "_gpu_free_mib", side_effect=lambda g: {0: 30000.0, 1: 100.0}[g]):
            return DynamicRouter._by_free_vram(self.slots)[0].gpu_id

    def test_guaranteed_card_is_chosen_even_when_occupied(self):
        self.assertEqual(1, self._first())

    def test_an_unreadable_guaranteed_card_sorts_last(self):
        with cards(), \
                patch.object(mm, "_gpu_free_mib", side_effect=lambda g: {0: 100.0, 1: None}[g]):
            self.assertEqual(0, DynamicRouter._by_free_vram(self.slots)[0].gpu_id)

    def test_preference_stays_lost_past_the_scale_out_cooldown(self):
        self.assertGreater(mm.GPU_CLEAR_RETRY_SECONDS, DynamicRouter.SCALE_OUT_COOLDOWN)

    def test_a_card_that_could_not_be_cleared_loses_its_preference(self):
        mm._gpu_clear_failed_at["GPU-bbb"] = mm.time.monotonic()
        self.assertEqual(0, self._first())


class SignalTests(unittest.TestCase):
    def _send(self, started_now, own_user=False):
        with patch.object(mm, "_pid_started", return_value=started_now), \
                patch.object(mm.os, "kill", side_effect=None if own_user else PermissionError) as own, \
                patch.object(mm.subprocess, "run") as sudo:
            mm._signal_pid(1234, signal.SIGTERM, "100")
        return own, sudo

    def test_another_users_process_is_signalled_through_sudo(self):
        _own, sudo = self._send("100")
        self.assertEqual(["sudo", "-n", "kill", "-15", "1234"], sudo.call_args.args[0])

    def test_our_users_process_is_signalled_directly(self):
        own, sudo = self._send("100", own_user=True)
        own.assert_called_once_with(1234, signal.SIGTERM)
        sudo.assert_not_called()

    def test_a_reused_pid_is_not_signalled(self):
        own, sudo = self._send("200")
        own.assert_not_called()
        sudo.assert_not_called()

    def test_the_start_time_of_a_live_process_is_read(self):
        self.assertTrue(mm._pid_started(mm.os.getpid()).isdigit())
        self.assertIsNone(mm._pid_started(2**30))


class DescribeTests(unittest.TestCase):
    def test_no_argument_of_another_process_reaches_the_log(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", "secret-token"])
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        text = mm._describe_pid(child.pid)
        self.assertNotIn("secret-token", text)
        self.assertNotIn("sleep", text)
        self.assertIn("python", text)


if __name__ == "__main__":
    unittest.main()
