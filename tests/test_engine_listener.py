"""The engine key is sent to a slot port only when the engine we started is what listens there."""
from __future__ import annotations

import socket
import subprocess
import sys
import unittest
from unittest import mock

import model_manager as mm

LISTEN = ("import socket,sys,time; s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
          "print(s.getsockname()[1], flush=True); time.sleep(30)")


class ListenerTests(unittest.TestCase):
    def setUp(self):
        self.child = subprocess.Popen([sys.executable, "-c", LISTEN], stdout=subprocess.PIPE, text=True,
                                      start_new_session=True)
        self.addCleanup(self.child.wait)
        self.addCleanup(self.child.kill)
        self.addCleanup(self.child.stdout.close)
        self.port = int(self.child.stdout.readline())

    def test_only_the_engines_own_process_group_counts(self):
        self.assertTrue(mm._listener_is_ours(self.port, self.child.pid))
        self.assertFalse(mm._listener_is_ours(self.port, self.child.pid + 1))
        self.assertFalse(mm._listener_is_ours(self.port, None))

    def test_a_port_nothing_listens_on_does_not_count(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            free = s.getsockname()[1]
        self.assertFalse(mm._listener_is_ours(free, self.child.pid))


class SpawnTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_foreign_listener_fails_the_start_with_that_reason(self):
        backend = mm.GpuBackend("qwen3.8-27b", "run.sh", "qwen3.8-27b", mm.GpuSlot(0, 0, 9000))
        process = mock.Mock(pid=4242, poll=mock.Mock(return_value=None))
        healthy = mock.MagicMock()
        healthy.__aenter__ = mock.AsyncMock(return_value=mock.Mock(status=200))
        healthy.__aexit__ = mock.AsyncMock(return_value=False)
        backend._session = mock.Mock(get=mock.Mock(return_value=healthy))
        with mock.patch("builtins.open", mock.mock_open()), mock.patch.object(mm.subprocess, "Popen", return_value=process), \
                mock.patch.object(mm, "_listener_is_ours", return_value=False), \
                mock.patch.object(backend, "_kill_process_locked", mock.AsyncMock()) as kill, \
                mock.patch.object(mm, "HEALTH_POLL", 0), mock.patch.object(mm.os, "killpg") as killpg, \
                self.assertRaisesRegex(RuntimeError, "not the engine just started"):
            await backend._spawn_attempt_locked(None)
        kill.assert_awaited_once()
        killpg.assert_not_called()


if __name__ == "__main__":
    unittest.main()
