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


if __name__ == "__main__":
    unittest.main()
