"""VISION_GPU_IDS: only some instances take images; images never cross regions."""
from __future__ import annotations

import unittest
from unittest import mock

import model_manager as mm
from tests.test_model_manager_context_fit import MODEL, RouterHarness

IMAGE = {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}


def body(*parts):
    return {"model": MODEL, "messages": [{"role": "user", "content": list(parts) or "hi"}]}


class ParseTests(unittest.TestCase):
    def test_values(self):
        self.assertIsNone(mm._parse_vision_gpus(""))
        self.assertEqual(frozenset(), mm._parse_vision_gpus("none"))
        self.assertEqual(frozenset({0, 2}), mm._parse_vision_gpus("0, 2"))

    def test_inline_only_flag(self):
        for spec, expected in (("", False), ("0", False), ("1", True)):
            self.assertIs(expected, mm._parse_flag("X", spec))
        with self.assertRaises(ValueError):
            mm._parse_flag("X", "yes")


class ImageRoutingTests(RouterHarness):
    async def _setup(self, vision_gpus):
        self.router.model_configs[MODEL] = mm.replace(
            self.router.model_configs[MODEL], vision_gpu_ids=vision_gpus)
        self.asked: list = []
        stub = self.router._get_or_start

        async def get_or_start(model, gpus=None):
            self.asked.append(gpus)
            return await stub(model)

        self.router._get_or_start = get_or_start

    async def _post(self, payload, **headers):
        return await self.client.post("/v1/chat/completions", json=payload, headers=headers)

    async def test_an_image_goes_to_a_vision_gpu_and_text_goes_anywhere(self):
        await self._setup(frozenset({0}))
        self.assertEqual(200, (await self._post(body(IMAGE))).status)
        self.assertEqual(200, (await self._post(body())).status)
        self.assertEqual([frozenset({0}), None], self.asked)

    async def test_an_image_is_never_overflowed(self):
        await self._setup(frozenset({0}))
        self.router._queue_waiting[MODEL] = 99
        backend = (await self.router._get_or_start(MODEL))[0]
        backend.waiting_behind = lambda priority: 0
        self.asked.clear()
        with mock.patch.object(mm, "OVERFLOW_QUEUE_DEPTH", 4):
            self.assertEqual(200, (await self._post(body(IMAGE))).status)
            self.assertEqual(503, (await self._post(body())).status)

    async def test_a_host_without_vision_answers_400_without_starting_anything(self):
        await self._setup(frozenset())
        response = await self._post(body(IMAGE))
        self.assertEqual(400, response.status)
        self.assertEqual([], self.asked)
        self.assertEqual(200, (await self._post(body())).status)

    async def test_handed_over_images_are_refused(self):
        header = {mm.OVERFLOW_HEADER: "1"}
        await self._setup(frozenset())   # peer's image arriving here: retryable there
        self.assertEqual(503, (await self._post(body(IMAGE), **header)).status)
        await self._setup(frozenset({0}))   # peer region takes no images
        self.assertEqual(400, (await self._post(body(IMAGE), **header)).status)
        self.assertEqual([], self.forwarded)

    async def test_responses_input_image_is_recognised(self):
        await self._setup(frozenset())
        response = await self.client.post("/v1/responses", json={"model": MODEL, "input": [
            {"role": "user", "content": [{"type": "input_image", "image_url": "https://example.com/a.png"}]}]})
        self.assertEqual(400, response.status)

    async def test_an_image_may_only_be_inline_or_on_the_web(self):
        await self._setup(None)
        for url in ("/etc/passwd", "file:///etc/passwd", "ftp://host/a.png", "a.png", 7):
            for image in ({"url": url}, url):
                payload = body({"type": "image_url", "image_url": image})
                self.assertEqual(400, (await self._post(payload)).status, image)
        self.assertEqual([], self.forwarded)
        for url in ("data:image/png;base64,AA==", "HTTPS://example.com/a.png"):
            self.assertEqual(200, (await self._post(body({"type": "image_url", "image_url": {"url": url}}))).status)

    async def test_an_inline_only_host_takes_nothing_but_base64_data_urls(self):
        await self._setup(frozenset({0}))
        self.router.model_configs[MODEL] = mm.replace(self.router.model_configs[MODEL], inline_images_only=True)
        for url in ("https://example.com/a.png", "http://127.0.0.1:1/a.png", "data:image/png,AA==",
                    "DATA:image/png;base64,AA=="):
            for image in ({"url": url}, url):
                payload = body({"type": "image_url", "image_url": image})
                self.assertEqual(400, (await self._post(payload)).status, image)
        self.assertEqual([], self.asked)
        inline = body({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}})
        self.assertEqual(200, (await self._post(inline)).status)
        self.assertEqual([frozenset({0})], self.asked)

    async def test_a_handed_over_inline_image_stays_retryable_on_a_vision_host(self):
        """The peer refused it for lack of capacity, not for what it is: a 400 would tell its caller to give up."""
        await self._setup(frozenset({0}))
        inline = body({"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}})
        self.assertEqual(503, (await self._post(inline, **{mm.OVERFLOW_HEADER: "1"})).status)
        self.assertEqual([], self.forwarded)

    async def test_a_part_without_a_type_may_not_name_a_server_path(self):
        await self._setup(None)
        for part in ({"image_url": {"url": "file:///etc/passwd"}}, {"type": "text", "uuid": "u", "image_url": "file:///x"}):
            self.assertEqual(400, (await self._post(body(part))).status, part)
        self.assertEqual([], self.forwarded)

    async def test_a_part_without_a_type_is_still_an_image(self):
        """vLLM takes an `image_url` key with no type (or with a uuid) as an image."""
        await self._setup(frozenset({0}))
        self.assertEqual(200, (await self._post(body({"image_url": {"url": "https://example.com/a.png"}}))).status)
        self.assertEqual([frozenset({0})], self.asked)

    async def test_an_inline_only_host_takes_no_video_or_audio(self):
        """Their URLs would be loaded by the engine as image URLs would, and none of its models take them."""
        await self._setup(frozenset({0}))
        self.router.model_configs[MODEL] = mm.replace(self.router.model_configs[MODEL], inline_images_only=True)
        for part in ({"type": "video_url", "video_url": {"url": "file:///x.mp4"}},
                     {"type": "audio_url", "audio_url": {"url": "https://example.com/a.wav"}},
                     {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}},
                     {"video_url": {"url": "data:video/mp4;base64,AA=="}},
                     {"type": "text", "audio_url": {"url": "data:audio/wav;base64,AA=="}},
                     {"type": "bogus", "video": "data:,"}):
            response = await self._post(body(part))
            self.assertEqual(400, response.status, part)
        self.assertEqual([], self.forwarded)

    async def test_only_content_parts_count_as_video_or_audio(self):
        """An SDK replays an assistant message with `"audio": null`, or with a reference to its audio."""
        await self._setup(frozenset({0}))
        self.router.model_configs[MODEL] = mm.replace(self.router.model_configs[MODEL], inline_images_only=True)
        history = [{"role": "user", "content": "hi", "audio": None},
                   {"role": "assistant", "content": [{"type": "text", "text": "x", "video": None}], "audio": None},
                   {"role": "assistant", "content": "y", "audio": {"id": "audio_1"}},
                   {"role": "user", "content": "again"}]
        response = await self._post({"model": MODEL, "messages": history})
        self.assertEqual(200, response.status)
        response = await self.client.post("/v1/responses", json={"model": MODEL, "input": [
            {"type": "input_audio", "input_audio": {"data": "AA==", "format": "wav"}}]})
        self.assertEqual(400, response.status)

    async def test_unrestricted_models_are_unchanged(self):
        await self._setup(None)
        self.assertEqual(200, (await self._post(body(IMAGE), **{mm.OVERFLOW_HEADER: "1"})).status)


class VisionSlotTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        config = mm.ModelConfig("run.sh", MODEL, max_num_seqs=8, vision_gpu_ids=frozenset({0}))
        other = mm.ModelConfig("other.sh", "other", max_num_seqs=8)
        self.slots = [mm.GpuSlot(0, 0, 9000), mm.GpuSlot(1, 1, 9010)]
        self.router = mm.DynamicRouter(self.slots, {MODEL: config, "other": other})

    def test_each_instance_is_told_its_vision_mode(self):
        self.assertTrue(self.router._make_backend(MODEL, self.slots[0]).vision)
        self.assertFalse(self.router._make_backend(MODEL, self.slots[1]).vision)
        self.assertIsNone(self.router._make_backend("other", self.slots[1]).vision)

    async def test_an_image_starts_the_vision_gpu_beside_a_text_only_instance(self):
        text_only = mock.Mock(model_name=MODEL, gpu_id=1, _ready=True, _closing=False, _failed=False,
                              is_running=True, slot=self.slots[1])
        self.slots[1].backend = text_only

        async def start(backend):
            backend._ready = True

        with mock.patch.object(mm.GpuBackend, "start", mock.AsyncMock()), \
                mock.patch.object(mm.GpuBackend, "_ensure_running", start), \
                mock.patch.object(mm.GpuBackend, "is_running", True), \
                mock.patch.object(mm, "_gpu_free_mib", return_value=1000.0):
            got = await self.router._get_or_start(MODEL, frozenset({0}))
        self.assertEqual([0], [b.gpu_id for b in got])

    async def test_a_vision_gpu_held_by_another_model_is_busy(self):
        self.slots[0].backend = mock.Mock(model_name="other", _failed=False, is_running=True)
        with self.assertRaises(mm.GPUBusyError):
            await self.router._get_or_start(MODEL, frozenset({0}))


class AdoptionTests(unittest.TestCase):
    def _adopts(self, extra, gpu_id, vision_gpus=frozenset({0})):
        args = ["vllm", "serve", "--max-num-seqs", "8", "--gpu-memory-utilization", "0.9", *extra]
        config = mm.ModelConfig("run.sh", MODEL, max_num_seqs=8, vision_gpu_ids=vision_gpus)
        with mock.patch.object(mm, "_read_vllm_cmdline", return_value=args):
            return mm._adopted_vllm_matches_config(1, "qwen3.8-27b", config, gpu_id)

    def test_only_the_right_vision_mode_is_adopted(self):
        self.assertTrue(self._adopts([], 0))
        self.assertFalse(self._adopts([], 1))
        self.assertTrue(self._adopts(["--language-model-only"], 1))
        self.assertFalse(self._adopts(["--language-model-only"], 0))
        self.assertTrue(self._adopts([], 1, vision_gpus=None))


class ScriptTests(unittest.TestCase):
    def test_the_vllm_script_honours_the_switch(self):
        with open(f"{mm.SCRIPT_DIR}/run_qwen38_27b.sh") as f:
            text = f.read()
        self.assertIn("QWEN_VISION", text)
        self.assertIn("--language-model-only", text)


if __name__ == "__main__":
    unittest.main()
