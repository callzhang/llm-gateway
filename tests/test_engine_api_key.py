"""The engines take model_manager's key and nobody else's."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import model_manager
from model_manager import DynamicRouter, GpuSlot, ModelConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine_plugins"))
from require_api_key import RequireApiKey  # noqa: E402

KEY = "k" * 64
VLLM_SCRIPTS = (
    "run_qwen38_27b.sh", "run_qwen36_35b_heretic.sh",
    "run_qwen3_asr_0_6b.sh", "run_qwen3_tts_1_7b_customvoice.sh",
)
VLLM_ARGS = ["vllm", "serve", "m", "--served-model-name", "chat-a",
             "--middleware", "require_api_key.RequireApiKey"]
SGLANG_ARGS = ["python", "-m", "sglang.launch_server", "--config", "/x/engine-api-key.yaml"]


def _router():
    return DynamicRouter(
        [GpuSlot(0, 0, 1)], {"chat-a": ModelConfig("run_chat.sh", "chat-a", max_num_seqs=4)}, None,
    )


class ForwardedKeyTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_clients_authorization_is_replaced_by_the_engine_key(self):
        seen = []

        async def engine(request):
            seen.append(request.headers.getall("Authorization", []))
            return web.json_response({})

        app = web.Application()
        app.router.add_post("/v1/chat/completions", engine)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        backend = _router()._make_backend("chat-a", GpuSlot(0, 0, server.port))
        self.addAsyncCleanup(backend._close_session)
        backend._ensure_session()

        front = web.Application()

        async def proxy(request):
            return await backend._forward(request, await request.read())

        front.router.add_post("/v1/chat/completions", proxy)
        client = TestClient(TestServer(front))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        with patch.object(model_manager, "ENGINE_API_KEY", KEY):
            for headers in ({"Authorization": "Bearer client"}, {"authorization": "Bearer client"}, {}):
                response = await client.post("/v1/chat/completions", json={}, headers=headers)
                self.assertEqual(200, response.status)
        self.assertEqual([[f"Bearer {KEY}"]] * 3, seen)


class MiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def _call(self, path, auth=None, kind="http"):
        reached, sent = [], []

        async def app(scope, receive, send):
            reached.append(scope["path"])

        async def send(message):
            sent.append(message)

        with patch.dict(os.environ, {"ENGINE_API_KEY": KEY}):
            guard = RequireApiKey(app)
        headers = [(b"authorization", auth.encode())] if auth else []
        await guard({"type": kind, "path": path, "headers": headers}, AsyncMock(), send)
        return bool(reached), sent

    async def test_every_route_but_the_probes_needs_the_key(self):
        for path in ("/v1/chat/completions", "/invocations", "/tokenize",
                     "/scale_elastic_ep", "/healthz", "/metrics/x"):
            for auth in (None, "Bearer local-qwen36", KEY):
                reached, sent = await self._call(path, auth)
                self.assertFalse(reached, (path, auth))
                self.assertEqual(401, sent[0]["status"])
            self.assertTrue((await self._call(path, f"Bearer {KEY}"))[0], path)
        for path in ("/health", "/metrics"):
            self.assertTrue((await self._call(path))[0], path)

    async def test_websockets_need_the_key_too(self):
        reached, sent = await self._call("/v1/realtime", kind="websocket")
        self.assertFalse(reached)
        self.assertEqual("websocket.close", sent[0]["type"])
        self.assertTrue((await self._call("/v1/realtime", f"Bearer {KEY}", "websocket"))[0])

    def test_it_does_not_start_without_a_key(self):
        for env in ({}, {"ENGINE_API_KEY": ""}):
            with patch.dict(os.environ, env, clear=True), self.assertRaises((KeyError, RuntimeError)):
                RequireApiKey(None)


class LaunchScriptTests(unittest.TestCase):
    def test_no_script_or_command_line_carries_a_key(self):
        for name in (*VLLM_SCRIPTS, "run_qwen38_27b_sglang.sh"):
            text = (ROOT / name).read_text()
            self.assertNotIn("local-qwen36", text, name)
            self.assertNotIn("--api-key", text, name)
            self.assertIn("${ENGINE_API_KEY:?", text, name)
        for name in VLLM_SCRIPTS:
            self.assertIn("--middleware require_api_key.RequireApiKey", (ROOT / name).read_text(), name)

    def test_a_vllm_launcher_passes_the_key_only_through_the_environment(self):
        for name in VLLM_SCRIPTS:
            env = {**os.environ, "ENGINE_API_KEY": KEY, "VLLM_GPU_MEM_UTIL": "0.5",
                   "VLLM_MAX_NUM_SEQS": "4", "VLLM_BIN": "/usr/bin/env",
                   "VLLM_ASR_BIN": "/usr/bin/env", "VLLM_TTS_BIN": "/usr/bin/env"}
            # `env serve ...` fails, but bash -x shows the exec line it was given.
            trace = subprocess.run(["bash", "-x", str(ROOT / name)], env=env,
                                   capture_output=True, text=True).stderr
            exec_line = next(l for l in trace.splitlines() if l.startswith("+ exec "))
            self.assertIn("require_api_key.RequireApiKey", exec_line, name)
            self.assertNotIn(KEY, exec_line, name)

    def test_the_sglang_launcher_writes_the_key_to_a_private_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "run.sh"
            text = (ROOT / "run_qwen38_27b_sglang.sh").read_text()
            self.assertIn("exec /data/derek/venv-sglang0517/bin/python", text)
            script.write_text(text.replace("exec /data/derek/venv-sglang0517/bin/python", "printf '%s\\n'"))
            env = {**os.environ, "ENGINE_API_KEY": KEY, "VLLM_CUDA_DEVICE": "0", "VLLM_PORT": "1",
                   "VLLM_GPU_MEM_UTIL": "0.5", "VLLM_MAX_NUM_SEQS": "4"}
            run = subprocess.run([shutil.which("bash"), str(script)], env=env,
                                 capture_output=True, text=True, check=True)
            config = Path(tmp) / "engine-api-key.yaml"
            args = run.stdout.splitlines()
            self.assertEqual(str(config), args[args.index("--config") + 1])
            self.assertNotIn(KEY, run.stdout)
            self.assertEqual(f"api-key: '{KEY}'\n", config.read_text())
            self.assertEqual(0o600, config.stat().st_mode & 0o777)
            self.assertEqual(["engine-api-key.yaml", "run.sh"], sorted(os.listdir(tmp)))


class AdoptionTests(unittest.IsolatedAsyncioTestCase):
    def _accepts(self, args, key):
        with patch.object(model_manager, "ENGINE_API_KEY", KEY), \
                patch.object(model_manager, "_read_vllm_cmdline", return_value=args), \
                patch.object(model_manager, "_read_process_env", return_value=key):
            return model_manager._adopted_engine_has_gateway_key(123)

    def test_only_an_engine_started_with_the_current_key_is_accepted(self):
        for args in (VLLM_ARGS, SGLANG_ARGS):
            self.assertTrue(self._accepts(args, KEY))
            for stale in (None, "", "old-key"):
                self.assertFalse(self._accepts(args, stale))
            self.assertFalse(self._accepts([*args, "--api-key", "local-qwen36"], KEY))
        # Launched by the old scripts, with the variable merely inherited.
        self.assertFalse(self._accepts(["vllm", "serve", "m"], KEY))
        self.assertFalse(self._accepts(["python", "-m", "sglang.launch_server"], KEY))
        self.assertFalse(self._accepts(None, KEY))

    def test_the_environment_of_a_real_process_is_read(self):
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; print(flush=True); time.sleep(30)"],
            env={"ENGINE_API_KEY": KEY}, stdout=subprocess.PIPE,
        )
        self.addCleanup(child.stdout.close)
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        child.stdout.readline()   # running, past exec
        self.assertEqual(KEY, model_manager._read_process_env(child.pid, "ENGINE_API_KEY"))
        self.assertIsNone(model_manager._read_process_env(child.pid, "ENGINE_API"))
        self.assertIsNone(model_manager._read_process_env(2**30, "ENGINE_API_KEY"))

    async def test_a_restart_stops_an_engine_with_another_key_and_does_not_adopt_it(self):
        router = _router()
        slot = router.slots[0]
        with patch.object(model_manager, "_find_vllm_pid_for_port", return_value=123), \
                patch.object(model_manager, "_read_served_model_name", return_value="chat-a"), \
                patch.object(model_manager, "_adopted_engine_has_gateway_key", return_value=False), \
                patch.object(model_manager, "_adopted_vllm_matches_config", return_value=True), \
                patch.object(model_manager, "_terminate_adopted_vllm", new=AsyncMock()) as stop:
            await router._try_adopt_slot(slot, MagicMock())
        stop.assert_awaited_once_with(123)
        self.assertIsNone(slot.backend)


class StartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_the_manager_refuses_to_run_without_a_key(self):
        with patch.object(model_manager, "ENGINE_API_KEY", ""), self.assertRaises(SystemExit):
            await model_manager.main()


if __name__ == "__main__":
    unittest.main()
