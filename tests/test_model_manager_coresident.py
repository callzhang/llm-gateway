"""Co-resident small models: scheduled by VRAM budget, never by taking a GPU slot.

A co-resident model (ASR, ~3 GiB) shares a GPU with whatever primary model owns
the slot.  It is admitted only if the GPU's *current* free VRAM covers its
min-viable footprint, and it yields to primaries: an idle one is stopped when a
primary spawn would otherwise be too tight.
"""

import asyncio
import unittest

import numpy as np
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import model_manager
from asr_adapter.cpu_pool import CpuWorkerError
from asr_adapter.engine import EngineError
from model_manager import (
    DynamicRouter,
    GPUBusyError,
    GpuBackend,
    GpuSlot,
    ModelConfig,
)

ASR = "qwen3-asr-0.6b"
SMALL_B = "small-b"
CHAT = "chat-a"
TOTAL_MIB = 32000.0


def _configs(**extra):
    configs = {
        CHAT: ModelConfig("run_chat.sh", CHAT, max_num_seqs=4),
        ASR: ModelConfig(
            "run_asr.sh", ASR, request_kind="transcription",
            coresident=True, idle_timeout=300,
        ),
        SMALL_B: ModelConfig(
            "run_b.sh", SMALL_B, request_kind="transcription", coresident=True,
        ),
    }
    configs.update(extra)
    return configs


def _router(slots=None):
    slots = slots or [GpuSlot(0, 0, 9000), GpuSlot(1, 1, 9010)]
    return DynamicRouter(slots, _configs())


def _fake_backend(router, model, slot, *, ready=True, running=True, active=0,
                  pid=None, failed=False):
    b = router._make_backend(model, slot)
    b._ready = ready
    b._failed = failed
    b._active_requests = active
    if running:
        b.process = SimpleNamespace(pid=pid or 4242, poll=lambda: None)
    slot.backend = b
    return b


def _vram(free_by_gpu):
    return (
        patch.object(model_manager, "_gpu_free_mib", side_effect=lambda g: free_by_gpu.get(g)),
        patch.object(model_manager, "_gpu_total_mib", return_value=TOTAL_MIB),
    )


class ModelConfigTests(unittest.TestCase):
    def test_transcription_model_needs_no_seq_limit_but_chat_still_does(self):
        cfg = ModelConfig("a.sh", "a", request_kind="transcription", coresident=True)
        self.assertTrue(cfg.coresident)
        self.assertIsNone(cfg.idle_timeout)
        with self.assertRaises(ValueError):
            ModelConfig("a.sh", "a")  # chat without max_num_seqs

    def test_idle_timeout_must_be_a_positive_int(self):
        for bad in (0, -5, 1.5, "300"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                ModelConfig("a.sh", "a", request_kind="transcription", idle_timeout=bad)

    def test_registered_asr_model_is_coresident_with_short_idle_and_a_vram_budget(self):
        cfg = model_manager.MODEL_CONFIGS[ASR]
        self.assertEqual("transcription", cfg.request_kind)
        self.assertTrue(cfg.coresident)
        self.assertEqual(300, cfg.idle_timeout)
        self.assertIn(ASR, model_manager.MODEL_GPU_MEM_UTIL)
        self.assertLess(
            model_manager.MODEL_MIN_GPU_MEM_UTIL[ASR],
            model_manager.MODEL_GPU_MEM_UTIL[ASR],
        )


class BudgetPlacementTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.router = _router()
        self.floor_mib = (
            model_manager.MODEL_MIN_GPU_MEM_UTIL.get(ASR, model_manager.GPU_MEM_UTIL_FLOOR)
            * TOTAL_MIB + model_manager.CORESIDENT_OVERHEAD_MIB
        )

    def _pick(self, free, model=ASR):
        p1, p2 = _vram(free)
        with p1, p2:
            return self.router._pick_coresident_gpu(model)

    def test_lane_ports_and_slot_ids_are_deterministic_and_clear_of_primaries(self):
        lanes = {
            (m, g): self.router._lane_slot(m, g)
            for m in (ASR, SMALL_B) for g in (0, 1)
        }
        ports = [s.port for s in lanes.values()]
        ids = [s.slot_id for s in lanes.values()]
        self.assertEqual(len(ports), len(set(ports)))
        self.assertEqual(len(ids), len(set(ids)))
        for slot in self.router.slots:
            self.assertNotIn(slot.port, ports)
            self.assertNotIn(slot.slot_id, ids)
        # Ports are far enough from every primary port that an EngineCore IPC
        # socket (api_port + 2) can never collide.
        for port in ports:
            for slot in self.router.slots:
                self.assertGreater(abs(port - slot.port), 10)
        self.assertIs(lanes[(ASR, 0)], self.router._lane_slot(ASR, 0))

    def test_places_on_the_gpu_with_room_and_never_needs_a_free_slot(self):
        _fake_backend(self.router, CHAT, self.router.slots[0])  # GPU0 primary, tight
        self.assertEqual(1, self._pick({0: 1500.0, 1: 26000.0}))

    def test_coexists_next_to_a_primary_when_the_budget_fits(self):
        router = _router([GpuSlot(0, 0, 9000)])
        primary = _fake_backend(router, CHAT, router.slots[0])
        p1, p2 = _vram({0: self.floor_mib + 1000})
        with p1, p2:
            self.assertEqual(0, router._pick_coresident_gpu(ASR))
        self.assertIs(primary, router.slots[0].backend)

    def test_prefers_a_gpu_without_a_primary_over_more_free_vram(self):
        _fake_backend(self.router, CHAT, self.router.slots[0])
        self.assertEqual(1, self._pick({0: 20000.0, 1: self.floor_mib + 500}))

    def test_refuses_with_a_readable_reason_when_no_gpu_has_the_budget(self):
        with self.assertRaises(GPUBusyError) as caught:
            self._pick({0: 1500.0, 1: 2000.0})
        message = str(caught.exception)
        self.assertIn(ASR, message)
        self.assertIn("GiB", message)
        self.assertIn("GPU 0", message)
        self.assertIn("GPU 1", message)

    def test_boundary_needs_the_min_viable_footprint_plus_cuda_context(self):
        self.assertEqual(0, self._pick({0: self.floor_mib, 1: 0.0}))
        with self.assertRaises(GPUBusyError):
            self._pick({0: self.floor_mib - 1, 1: 0.0})

    def test_starting_coresident_models_reserve_their_budget_before_vram_shows_it(self):
        # ASR is mid-spawn on GPU0 (claimed, not ready): its eventual footprint
        # must be subtracted or a second model would oversubscribe the card.
        lane = self.router._lane_slot(ASR, 0)
        _fake_backend(self.router, ASR, lane, ready=False, running=False)
        reserved = model_manager.MODEL_GPU_MEM_UTIL[ASR] * TOTAL_MIB
        need_b = (
            model_manager.MODEL_MIN_GPU_MEM_UTIL.get(SMALL_B, model_manager.GPU_MEM_UTIL_FLOOR)
            * TOTAL_MIB + model_manager.CORESIDENT_OVERHEAD_MIB
        )
        router = _router([GpuSlot(0, 0, 9000)])
        lane = router._lane_slot(ASR, 0)
        _fake_backend(router, ASR, lane, ready=False, running=False)
        p1, p2 = _vram({0: reserved + need_b - 1})
        with p1, p2, self.assertRaises(GPUBusyError):
            router._pick_coresident_gpu(SMALL_B)
        p1, p2 = _vram({0: reserved + need_b})
        with p1, p2:
            self.assertEqual(0, router._pick_coresident_gpu(SMALL_B))

    def test_honours_allowed_gpu_ids(self):
        configs = _configs()
        configs[ASR] = ModelConfig(
            "run_asr.sh", ASR, request_kind="transcription", coresident=True,
            allowed_gpu_ids={0},
        )
        router = DynamicRouter([GpuSlot(0, 0, 9000), GpuSlot(1, 1, 9010)], configs)
        p1, p2 = _vram({0: self.floor_mib, 1: 30000.0})
        with p1, p2:
            self.assertEqual(0, router._pick_coresident_gpu(ASR))


class RouterIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.router = _router()

    async def test_get_or_start_claims_a_lane_not_a_slot(self):
        started = []

        async def fake_ensure(self_backend):
            self_backend._ready = True
            self_backend.process = SimpleNamespace(pid=4242, poll=lambda: None)
            started.append(self_backend)

        p1, p2 = _vram({0: 1500.0, 1: 26000.0})
        with p1, p2, patch.object(GpuBackend, "_ensure_running", fake_ensure), \
                patch.object(GpuBackend, "start", AsyncMock()):
            backends = await self.router._get_or_start(ASR)

        self.assertEqual(1, len(backends))
        b = backends[0]
        self.assertEqual(ASR, b.model_name)
        self.assertEqual(1, b.gpu_id)
        self.assertEqual(300, b.idle_timeout)
        self.assertTrue(b.coresident)
        self.assertTrue(all(s.backend is None for s in self.router.slots))
        # A second request reuses the running lane.
        self.assertEqual([b], await self.router._get_or_start(ASR))

    async def test_all_gpus_full_is_a_503_gpu_busy_not_a_slot_error(self):
        p1, p2 = _vram({0: 1500.0, 1: 2000.0})
        with p1, p2, self.assertRaises(GPUBusyError):
            await self.router._get_or_start(ASR)

    async def test_failed_spawn_releases_the_lane(self):
        async def boom(self_backend):
            raise RuntimeError("vLLM exited")

        p1, p2 = _vram({0: 1500.0, 1: 26000.0})
        with p1, p2, patch.object(GpuBackend, "_ensure_running", boom), \
                patch.object(GpuBackend, "start", AsyncMock()):
            with self.assertRaises(RuntimeError):
                await self.router._get_or_start(ASR)
        self.assertIsNone(self.router._lane_slot(ASR, 1).backend)

    def test_running_and_claimed_lookups_include_lanes(self):
        _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 1))
        self.assertEqual(1, len(self.router._running_backends(ASR)))
        self.assertEqual(1, len(self.router._claimed_backends(ASR)))
        self.assertEqual(2, len(self.router._free_slots()))  # lanes never consume slots

    def test_status_lists_coresident_lanes_separately_from_slots(self):
        _fake_backend(self.router, CHAT, self.router.slots[0])
        _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 1), active=2)
        status = self.router.status()
        self.assertEqual([0, 1], [s["slot_id"] for s in status["slots"]])
        self.assertEqual(1, len(status["coresident"]))
        entry = status["coresident"][0]
        self.assertEqual(ASR, entry["model"])
        self.assertEqual(1, entry["gpu_id"])
        self.assertEqual("ready", entry["state"])
        self.assertEqual(2, entry["active_requests"])
        self.assertEqual(300, entry["idle_timeout"])

    async def test_no_scale_out_for_coresident_models(self):
        _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 1))
        with patch.object(GpuBackend, "_ensure_running", AsyncMock()) as spawn:
            await self.router._maybe_scale_out(ASR)
        spawn.assert_not_awaited()
        self.assertEqual(1, len(self.router._running_backends(ASR)))


class SpawnUtilTests(unittest.TestCase):
    """A co-resident vLLM must leave room for its own CUDA context (measured ~0.5 GiB):
    vLLM refuses to start if free VRAM *after* the context is below util*total."""

    def _util(self, model, free, coresident):
        router = _router()
        b = router._make_backend(model, router._lane_slot(model, 1) if coresident else router.slots[1])
        with patch.object(model_manager, "_gpu_free_mib", return_value=free), \
                patch.object(model_manager, "_gpu_total_mib", return_value=TOTAL_MIB):
            return b._gpu_mem_util_for_spawn()

    def test_coresident_never_clamps_up_into_the_context_overhead(self):
        floor = model_manager.MODEL_MIN_GPU_MEM_UTIL[ASR] * TOTAL_MIB
        # enough for the floor plus the old 256 MiB margin, but not for the context
        with self.assertRaisesRegex(RuntimeError, "too tight"):
            self._util(ASR, floor + model_manager.HARD_MARGIN_MIB + 10, True)
        # with the context overhead available, the floor is granted
        self.assertEqual(
            model_manager.MODEL_MIN_GPU_MEM_UTIL[ASR],
            self._util(ASR, floor + model_manager.CORESIDENT_OVERHEAD_MIB, True),
        )

    def test_coresident_gets_its_preferred_util_when_there_is_plenty_of_room(self):
        self.assertEqual(model_manager.MODEL_GPU_MEM_UTIL[ASR], self._util(ASR, 30000.0, True))


class PrimaryYieldsTests(unittest.IsolatedAsyncioTestCase):
    """Primaries outrank co-resident models for VRAM."""

    def setUp(self):
        self.router = _router()
        self.primary = GpuBackend(CHAT, "run_chat.sh", CHAT, self.router.slots[1])
        self.primary.router = self.router
        need = (
            model_manager.MODEL_MIN_GPU_MEM_UTIL.get(CHAT, model_manager.GPU_MEM_UTIL_FLOOR)
            * TOTAL_MIB + model_manager.HARD_MARGIN_MIB
        )
        self.need = need

    def _asr(self, **kw):
        b = _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 1), **kw)
        b.stop = AsyncMock()
        return b

    async def test_idle_coresident_is_stopped_when_the_primary_would_be_too_tight(self):
        asr = self._asr()
        free = iter([self.need - 3000, self.need + 500])
        p1 = patch.object(model_manager, "_gpu_free_mib", side_effect=lambda g: next(free))
        p2 = patch.object(model_manager, "_gpu_total_mib", return_value=TOTAL_MIB)
        with p1, p2, patch("asyncio.sleep", new=AsyncMock()):
            await self.router.reclaim_coresident_for(self.primary)
        asr.stop.assert_awaited_once()

    async def test_busy_coresident_is_left_alone(self):
        asr = self._asr(active=1)
        p1, p2 = _vram({1: self.need - 3000})
        with p1, p2, patch("asyncio.sleep", new=AsyncMock()):
            await self.router.reclaim_coresident_for(self.primary)
        asr.stop.assert_not_awaited()

    async def test_nothing_is_stopped_when_the_primary_already_fits(self):
        asr = self._asr()
        p1, p2 = _vram({1: self.need + 2000})
        with p1, p2, patch("asyncio.sleep", new=AsyncMock()):
            await self.router.reclaim_coresident_for(self.primary)
        asr.stop.assert_not_awaited()

    async def test_other_gpus_coresident_models_are_untouched(self):
        other = _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 0))
        other.stop = AsyncMock()
        p1, p2 = _vram({1: self.need - 3000, 0: 1000.0})
        with p1, p2, patch("asyncio.sleep", new=AsyncMock()):
            await self.router.reclaim_coresident_for(self.primary)
        other.stop.assert_not_awaited()

    async def test_only_primaries_trigger_reclaim_during_spawn(self):
        self.router.reclaim_coresident_for = AsyncMock()
        primary, lane_b = self.primary, GpuBackend(
            ASR, "run_asr.sh", ASR, self.router._lane_slot(ASR, 1), coresident=True,
        )
        lane_b.router = self.router
        for backend in (primary, lane_b):
            with patch.object(GpuBackend, "_check_gpu_free", side_effect=RuntimeError("stop here")), \
                    patch.object(GpuBackend, "_ensure_session"), \
                    patch.object(GpuBackend, "_ensure_idle_task"):
                with self.assertRaises(RuntimeError):
                    await backend._spawn_locked()
        self.router.reclaim_coresident_for.assert_awaited_once_with(primary)


class LeftoverCheckTests(unittest.TestCase):
    """_check_gpu_free must tolerate live neighbours but still catch orphans."""

    def setUp(self):
        self.router = _router()
        self.backend = GpuBackend(CHAT, "run_chat.sh", CHAT, self.router.slots[1])
        self.backend.router = self.router

    def _check(self, vllm_pids, pgids):
        with patch.object(model_manager, "_gpu_vllm_pids", return_value=vllm_pids), \
                patch.object(GpuBackend, "_pgid_of", side_effect=lambda pid: pgids.get(pid)):
            self.backend._check_gpu_free()

    def test_a_running_coresident_neighbour_is_not_a_leftover(self):
        _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 1), pid=700)
        self._check([(701, "6900 MiB")], {701: 700})  # EngineCore child, ASR's pgid

    def test_an_unknown_vllm_process_is_still_a_leftover(self):
        _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 1), pid=700)
        with self.assertRaisesRegex(RuntimeError, "leftover vLLM"):
            self._check([(701, "6900 MiB"), (999, "30000 MiB")], {701: 700, 999: 999})

    def test_a_dead_backends_pgid_is_no_longer_trusted(self):
        _fake_backend(self.router, ASR, self.router._lane_slot(ASR, 1), pid=700, running=False)
        with self.assertRaisesRegex(RuntimeError, "leftover vLLM"):
            self._check([(701, "6900 MiB")], {701: 700})

    def test_without_a_router_any_vllm_process_still_blocks(self):
        self.backend.router = None
        with self.assertRaisesRegex(RuntimeError, "leftover vLLM"):
            self._check([(701, "6900 MiB")], {701: 700})


class IdleTimeoutTests(unittest.TestCase):
    def test_model_idle_timeout_overrides_the_global_default(self):
        slot = GpuSlot(0, 0, 9000)
        asr = GpuBackend(ASR, "run_asr.sh", ASR, slot, idle_timeout=300, coresident=True)
        chat = GpuBackend(CHAT, "run_chat.sh", CHAT, slot)
        self.assertEqual(300, asr._base_idle_timeout())
        self.assertEqual(model_manager.IDLE_TIMEOUT, chat._base_idle_timeout())


class MultipartRoutingTests(unittest.IsolatedAsyncioTestCase):
    BOUNDARY = "xyzBOUNDARYxyz"

    def _multipart(self, *, model=ASR, audio=b"ID3\x00\x01audio-bytes", model_first=True, extra=None):
        b = self.BOUNDARY.encode()
        extra_parts = b"".join(
            b"--" + b + f'\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
            for k, v in (extra or {}).items()
        )
        model_part = (
            b"--" + b + b'\r\nContent-Disposition: form-data; name="model"\r\n\r\n'
            + model.encode() + b"\r\n"
        )
        file_part = (
            b"--" + b + b'\r\nContent-Disposition: form-data; name="file"; filename="a.mp3"\r\n'
            b"Content-Type: audio/mpeg\r\n\r\n" + audio + b"\r\n"
        )
        tail = b"--" + b + b"--\r\n"
        parts = model_part + file_part if model_first else file_part + model_part
        return extra_parts + parts + tail

    def test_extracts_the_model_from_a_multipart_body_in_either_order(self):
        ctype = f"multipart/form-data; boundary={self.BOUNDARY}"
        for first in (True, False):
            with self.subTest(model_first=first):
                body = self._multipart(model_first=first)
                self.assertEqual(ASR, DynamicRouter._extract_model(body, ctype))

    def test_binary_audio_that_looks_like_a_model_field_is_not_mistaken_for_one(self):
        ctype = f"multipart/form-data; boundary={self.BOUNDARY}"
        audio = b'\r\nContent-Disposition: form-data; name="model"\r\n\r\nevil\r\n'
        body = self._multipart(audio=audio, model_first=False)
        self.assertEqual(ASR, DynamicRouter._extract_model(body, ctype))

    def test_json_bodies_and_garbage_still_work(self):
        self.assertEqual("m", DynamicRouter._extract_model(b'{"model": "m"}'))
        self.assertEqual("m", DynamicRouter._extract_model(b'{"model": "m"}', "application/json"))
        self.assertIsNone(DynamicRouter._extract_model(b"", "multipart/form-data; boundary=x"))
        self.assertIsNone(DynamicRouter._extract_model(b"not json"))
        self.assertIsNone(
            DynamicRouter._extract_model(b"--b\r\n", "multipart/form-data; boundary=b")
        )

    async def asyncSetUp(self):
        self.router = _router()
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", self.router.handle)
        self.client = TestClient(TestServer(app))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def _post(self, path="/v1/audio/transcriptions", body=None):
        return await self.client.post(
            path,
            data=body if body is not None else self._multipart(),
            headers={"Content-Type": f"multipart/form-data; boundary={self.BOUNDARY}"},
        )

    @staticmethod
    def _speech(seconds=70):
        t = np.arange(int(seconds * 16000)) / 16000
        return (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    def _lane(self, text="你好"):
        backend = SimpleNamespace(
            slot=SimpleNamespace(slot_id=101), _active_requests=0,
            transcribe_chunk=AsyncMock(return_value=text),
        )
        self.router._get_or_start = AsyncMock(return_value=[backend])
        return backend

    async def test_upload_is_chunked_and_every_chunk_goes_to_the_gpu_lane(self):
        backend = self._lane()
        with patch.object(model_manager.asr_service, "decode_audio", AsyncMock(return_value=self._speech())):
            response = await self._post(body=self._multipart(extra={"response_format": "verbose_json", "language": "zh"}))
        self.assertEqual(200, response.status)
        body = await response.json()
        self.assertEqual("gpu", body["engine"])
        self.assertEqual(3, len(body["segments"]))                     # 70 s -> three ~30 s chunks
        self.assertEqual(3, backend.transcribe_chunk.await_count)
        wav, language = backend.transcribe_chunk.await_args.args
        self.assertEqual(b"RIFF", wav[:4])
        self.assertEqual("zh", language)

    async def test_a_busy_gpu_finishes_the_whole_request_on_the_cpu_worker(self):
        self.router._get_or_start = AsyncMock(side_effect=GPUBusyError("no room: need 3.7 GiB"))
        cpu = SimpleNamespace(transcribe=AsyncMock(return_value="甲乙丙"))
        with patch.object(self.router, "_asr_cpu", return_value=cpu), \
                patch.object(model_manager.asr_service, "decode_audio", AsyncMock(return_value=self._speech())):
            response = await self._post(body=self._multipart(extra={"response_format": "verbose_json"}))
        self.assertEqual(200, response.status)
        body = await response.json()
        self.assertEqual("cpu", body["engine"])
        self.assertEqual(3, cpu.transcribe.await_count)
        self.router._get_or_start.assert_awaited_once_with(ASR)        # the GPU is not retried per chunk

    async def test_gpu_failure_mid_request_falls_back_for_the_rest(self):
        backend = self._lane()
        backend.transcribe_chunk = AsyncMock(side_effect=["一", EngineError("HTTP 500")])
        cpu = SimpleNamespace(transcribe=AsyncMock(return_value="二"))
        with patch.object(self.router, "_asr_cpu", return_value=cpu), \
                patch.object(model_manager.asr_service, "decode_audio", AsyncMock(return_value=self._speech())):
            response = await self._post(body=self._multipart(extra={"response_format": "verbose_json"}))
        body = await response.json()
        self.assertEqual("gpu+cpu", body["engine"])
        self.assertEqual("一二二", body["text"])

    async def test_cpu_worker_failure_is_a_503(self):
        self.router._get_or_start = AsyncMock(side_effect=GPUBusyError("no room"))
        cpu = SimpleNamespace(transcribe=AsyncMock(side_effect=CpuWorkerError("CPU worker did not start")))
        with patch.object(self.router, "_asr_cpu", return_value=cpu), \
                patch.object(model_manager.asr_service, "decode_audio", AsyncMock(return_value=self._speech(10))):
            response = await self._post()
        self.assertEqual(503, response.status)
        self.assertEqual("service_unavailable", (await response.json())["error"]["type"])

    async def test_bad_uploads_are_client_errors_and_never_start_a_gpu(self):
        self.router._get_or_start = AsyncMock()
        bad_format = await self._post(body=self._multipart(extra={"response_format": "srt"}))
        self.assertEqual(400, bad_format.status)
        with patch.object(model_manager.asr_service, "decode_audio",
                          AsyncMock(side_effect=model_manager.asr_service.UploadError("audio could not be decoded"))):
            undecodable = await self._post()
        self.assertEqual(400, undecodable.status)
        self.router._get_or_start.assert_not_awaited()

    async def test_transcription_model_on_other_routes_is_rejected_before_gpu_start(self):
        self.router._get_or_start = AsyncMock()
        for path in ("/v1/chat/completions", "/v1/audio/speech", "/v1/completions"):
            with self.subTest(path=path):
                response = await self._post(path=path)
                self.assertEqual(400, response.status)
        self.router._get_or_start.assert_not_awaited()

    async def test_unknown_model_in_a_multipart_upload_is_404(self):
        self.router._get_or_start = AsyncMock()
        response = await self._post(body=self._multipart(model="nope"))
        self.assertEqual(404, response.status)
        self.router._get_or_start.assert_not_awaited()


class BackendTranscribeChunkTests(unittest.IsolatedAsyncioTestCase):
    """GpuBackend.transcribe_chunk: one chunk through vLLM, EngineError on any failure."""

    async def _serve(self, handler):
        app = web.Application()
        app.router.add_post("/v1/audio/transcriptions", handler)
        server = TestServer(app)
        await server.start_server()
        self.addAsyncCleanup(server.close)
        backend = _router()._make_backend(ASR, GpuSlot(101, 0, server.port))
        self.addAsyncCleanup(backend._close_session)
        return backend

    async def test_posts_a_wav_with_the_served_name_and_key_and_returns_the_text(self):
        seen = {}

        async def handler(request):
            form = await request.post()
            seen["auth"] = request.headers.get("Authorization")
            seen["model"], seen["lang"] = form["model"], form.get("language")
            seen["file"] = form["file"].file.read()
            return web.json_response({"text": "你好"})

        backend = await self._serve(handler)
        self.assertEqual("你好", await backend.transcribe_chunk(b"RIFFwav", "zh"))
        self.assertEqual(f"Bearer {model_manager.ASR_GPU_API_KEY}", seen["auth"])
        self.assertEqual((backend.served_name, "zh", b"RIFFwav"), (seen["model"], seen["lang"], seen["file"]))
        self.assertEqual(0, backend._active_requests)

    async def test_http_errors_become_engine_errors(self):
        async def handler(request):
            return web.json_response({"error": "overloaded"}, status=503)

        backend = await self._serve(handler)
        with self.assertRaises(EngineError):
            await backend.transcribe_chunk(b"x", None)
        self.assertEqual(0, backend._active_requests)

    async def test_an_unreachable_backend_is_an_engine_error(self):
        backend = _router()._make_backend(ASR, GpuSlot(101, 0, 1))      # nothing listens on port 1
        self.addAsyncCleanup(backend._close_session)
        with self.assertRaises(EngineError):
            await backend.transcribe_chunk(b"x", None)


class AdoptionTests(unittest.IsolatedAsyncioTestCase):
    async def test_running_coresident_vllm_is_adopted_after_a_manager_restart(self):
        router = _router()
        adopted = []

        async def fake_adopt(slot, session):
            adopted.append((slot.slot_id, slot.gpu_id, slot.port))

        lane_port = router._lane_slot(ASR, 1).port

        def find_pid(port):
            return 555 if port == lane_port else None

        with patch.object(model_manager, "_find_vllm_pid_for_port", side_effect=find_pid), \
                patch.object(router, "_try_adopt_slot", side_effect=fake_adopt):
            await router.adopt_existing_backends()

        lane = router._lane_slot(ASR, 1)
        self.assertIn((lane.slot_id, 1, lane_port), adopted)
        # Primary slots are still probed; empty lane ports are not.
        self.assertEqual(
            {0, 1}, {sid for sid, _, _ in adopted if sid in (0, 1)},
        )
        self.assertEqual(1, len([a for a in adopted if a[0] >= 100]))


if __name__ == "__main__":
    unittest.main()
