import asyncio
import json
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestClient, TestServer
import yaml

import model_manager


MODEL_NAME = "qwen3-tts-1.7b-customvoice"
REPO_ROOT = Path(__file__).resolve().parents[1]


class SpeechRoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = model_manager.ModelConfig(
            script="run_qwen3_tts_1_7b_customvoice.sh",
            served_name=MODEL_NAME,
            allowed_gpu_ids=None,
            request_kind="speech",
            max_input_chars=12,
        )
        self.router = model_manager.DynamicRouter(
            [model_manager.GpuSlot(0, 0, 9000)],
            {MODEL_NAME: self.config},
        )

    async def asyncSetUp(self):
        self.app = web.Application()
        self.app.router.add_route("*", "/{path_info:.*}", self.router.handle)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def _post(self, path="/v1/audio/speech", **overrides):
        payload = {
            "model": MODEL_NAME,
            "input": "你好",
            "voice": "Vivian",
            "instructions": "温暖、自然",
            "response_format": "mp3",
        }
        payload.update(overrides)
        return await self.client.post(path, json=payload)

    async def test_rejects_missing_blank_and_over_limit_input_before_gpu_start(self):
        self.router._get_or_start = AsyncMock()

        for value in (None, "", "   ", "十三个字符的输入文本超过限制"):
            with self.subTest(value=value):
                response = await self._post(input=value)
                self.assertEqual(400, response.status)
                body = await response.json()
                self.assertEqual("invalid_request_error", body["error"]["type"])

        self.router._get_or_start.assert_not_awaited()

    async def test_rejects_speech_model_on_non_speech_route_before_gpu_start(self):
        self.router._get_or_start = AsyncMock()
        response = await self._post(path="/v1/chat/completions")

        self.assertEqual(400, response.status)
        self.router._get_or_start.assert_not_awaited()

    async def test_all_requested_formats_are_normalized_to_mp3(self):
        for response_format in ("wav", "pcm", "opus", "flac", "mp3"):
            with self.subTest(response_format=response_format):
                backend = SimpleNamespace(
                    slot=SimpleNamespace(slot_id=0),
                    _active_requests=0,
                    proxy=AsyncMock(
                        return_value=web.Response(
                            body=b"ID3-test", content_type="audio/mpeg"
                        )
                    ),
                )
                self.router._get_or_start = AsyncMock(return_value=[backend])
                response = await self._post(response_format=response_format)
                self.assertEqual(200, response.status)
                forwarded_body = json.loads(backend.proxy.await_args.args[1])
                self.assertEqual("mp3", forwarded_body["response_format"])

    async def test_missing_response_format_defaults_to_mp3_before_forwarding(self):
        backend = SimpleNamespace(
            slot=SimpleNamespace(slot_id=0),
            _active_requests=0,
            proxy=AsyncMock(
                return_value=web.Response(body=b"ID3-test", content_type="audio/mpeg")
            ),
        )
        self.router._get_or_start = AsyncMock(return_value=[backend])

        response = await self.client.post(
            "/v1/audio/speech",
            json={"model": MODEL_NAME, "input": "你好", "voice": "Vivian"},
        )

        self.assertEqual(200, response.status)
        forwarded_body = json.loads(backend.proxy.await_args.args[1])
        self.assertEqual("mp3", forwarded_body["response_format"])

    async def test_valid_request_reaches_backend_without_logging_text(self):
        audio = b"ID3-test-mp3"
        slot = SimpleNamespace(slot_id=0)
        backend = SimpleNamespace(
            slot=slot,
            _active_requests=0,
            proxy=AsyncMock(return_value=web.Response(body=audio, content_type="audio/mpeg")),
        )
        self.router._get_or_start = AsyncMock(return_value=[backend])

        with self.assertLogs("mgr.router", level="INFO") as captured:
            response = await self._post()
            body = await response.read()

        self.assertEqual(200, response.status)
        self.assertEqual(audio, body)
        self.assertEqual("audio/mpeg", response.headers["Content-Type"])
        log_text = "\n".join(captured.output)
        self.assertNotIn("你好", log_text)
        self.assertNotIn("温暖、自然", log_text)
        self.assertIn("input_chars=2", log_text)


class BinaryForwardingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.audio = b"ID3\x04\x00\x00\xffMP3binary"

        async def speech(_request):
            return web.Response(body=self.audio, content_type="audio/mpeg")

        upstream_app = web.Application()
        upstream_app.router.add_post("/v1/audio/speech", speech)
        self.upstream = TestServer(upstream_app)
        await self.upstream.start_server()

        slot = model_manager.GpuSlot(0, 0, self.upstream.port)
        self.backend = model_manager.GpuBackend(
            MODEL_NAME,
            "run_qwen3_tts_1_7b_customvoice.sh",
            MODEL_NAME,
            slot,
        )
        self.backend._session = ClientSession()

        async def proxy(request):
            return await self.backend._forward(request, await request.read())

        proxy_app = web.Application()
        proxy_app.router.add_post("/v1/audio/speech", proxy)
        self.client = TestClient(TestServer(proxy_app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        await self.backend._session.close()
        await self.upstream.close()

    async def test_forwards_mp3_content_type_and_exact_binary_body(self):
        response = await self.client.post(
            "/v1/audio/speech",
            json={
                "model": MODEL_NAME,
                "input": "你好",
                "voice": "Vivian",
                "response_format": "mp3",
            },
        )

        self.assertEqual(200, response.status)
        self.assertEqual("audio/mpeg", response.headers["Content-Type"])
        self.assertEqual(self.audio, await response.read())


class SpeechLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_active_request_is_held_until_stream_finishes(self):
        backend = model_manager.GpuBackend(
            MODEL_NAME,
            "run_qwen3_tts_1_7b_customvoice.sh",
            MODEL_NAME,
            model_manager.GpuSlot(0, 0, 9000),
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked_forward(_request, _body):
            entered.set()
            await release.wait()
            return web.Response(body=b"done")

        with patch.object(backend, "_forward", side_effect=blocked_forward):
            task = asyncio.create_task(backend.proxy(None, b""))
            await entered.wait()
            self.assertEqual(1, backend._active_requests)
            release.set()
            await task

        self.assertEqual(0, backend._active_requests)


class TtsRegistrationTests(unittest.TestCase):
    def test_customvoice_model_is_registered_as_speech(self):
        config = model_manager.MODEL_CONFIGS[MODEL_NAME]
        self.assertEqual("speech", config.request_kind)
        self.assertGreater(config.max_input_chars, 0)

    def test_input_limit_is_configurable(self):
        with patch.dict(os.environ, {"QWEN3_TTS_MAX_INPUT_CHARS": "4321"}):
            self.assertEqual(4321, model_manager._tts_max_input_chars())

    def test_tts_requires_measured_free_vram_before_start(self):
        self.assertEqual(30.3, model_manager.MODEL_MIN_FREE_GIB[MODEL_NAME])


class LauncherContractTests(unittest.TestCase):
    def test_launcher_uses_dedicated_pinned_runtime_and_dynamic_slot(self):
        launcher = (
            REPO_ROOT / "run_qwen3_tts_1_7b_customvoice.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(".venv-tts/bin/vllm-omni", launcher)
        self.assertIn("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", launcher)
        self.assertIn("CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICE:-0}", launcher)
        self.assertIn("--host 127.0.0.1", launcher)
        self.assertIn("--port ${VLLM_PORT:-9000}", launcher)
        self.assertIn("--served-model-name qwen3-tts-1.7b-customvoice", launcher)
        self.assertIn("--deploy-config", launcher)
        self.assertIn("configs/qwen3_tts.yaml", launcher)
        self.assertIn("--omni", launcher)

    def test_deploy_config_keeps_both_stages_on_remapped_single_gpu(self):
        deploy_config = (REPO_ROOT / "configs/qwen3_tts.yaml").read_text(
            encoding="utf-8"
        )

        self.assertEqual(2, deploy_config.count("stage_id:"))
        self.assertEqual(2, deploy_config.count('devices: "0"'))
        self.assertEqual(2, deploy_config.count("gpu_memory_utilization: 0.3"))
        self.assertRegex(
            deploy_config,
            r"stage_id: 1[\s\S]*?max_num_seqs: 1(?:\D|$)",
        )


class LiteLlmConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = yaml.safe_load(
            (REPO_ROOT / "config.yaml").read_text(encoding="utf-8")
        )

    def test_tts_uses_existing_gateway_and_hosted_vllm_speech_provider(self):
        matches = [
            item for item in self.config["model_list"]
            if item["model_name"] == MODEL_NAME
        ]
        self.assertEqual(1, len(matches))
        params = matches[0]["litellm_params"]
        self.assertEqual(f"hosted_vllm/{MODEL_NAME}", params["model"])
        self.assertEqual("http://127.0.0.1:8002/v1", params["api_base"])
        self.assertEqual("local-qwen36", params["api_key"])
        self.assertEqual(600, params["timeout"])

    def test_voice_design_and_clone_models_are_not_exposed(self):
        names = {item["model_name"].lower() for item in self.config["model_list"]}
        self.assertFalse(any("voicedesign" in name for name in names))
        self.assertNotIn("qwen3-tts-1.7b-base", names)

    def test_open_webui_tts_uses_litellm_and_scoped_key(self):
        launcher = (REPO_ROOT / "run_open_webui.sh").read_text(encoding="utf-8")

        self.assertIn("export AUDIO_TTS_ENGINE=openai", launcher)
        self.assertIn(
            'export AUDIO_TTS_OPENAI_API_BASE_URL="http://127.0.0.1:8900/v1"',
            launcher,
        )
        self.assertIn(
            'export AUDIO_TTS_OPENAI_API_KEY="$OPENWEBUI_LLM_KEY"', launcher
        )
        self.assertIn(f"export AUDIO_TTS_MODEL={MODEL_NAME}", launcher)
        self.assertIn("export AUDIO_TTS_VOICE=Vivian", launcher)


if __name__ == "__main__":
    unittest.main()
