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
