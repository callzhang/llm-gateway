"""Co-resident small models: scheduled by VRAM budget, never by taking a GPU slot.

A co-resident model (ASR, ~3 GiB) shares a GPU with whatever primary model owns
the slot.  It is admitted only if the GPU's *current* free VRAM covers its
min-viable footprint, and it yields to primaries: an idle one is stopped when a
primary spawn would otherwise be too tight.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import model_manager
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
            * TOTAL_MIB + model_manager.HARD_MARGIN_MIB
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

    def test_boundary_needs_the_min_viable_footprint_plus_margin(self):
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
            * TOTAL_MIB + model_manager.HARD_MARGIN_MIB
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

    def _multipart(self, *, model=ASR, audio=b"ID3\x00\x01audio-bytes", model_first=True):
        b = self.BOUNDARY.encode()
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
        return parts + tail

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

    async def test_forwards_the_upload_byte_for_byte_with_its_content_type(self):
        seen = {}

        async def proxy(request, body):
            seen["body"], seen["ctype"] = body, request.headers["Content-Type"]
            return web.json_response({"text": "ok"})

        backend = SimpleNamespace(slot=SimpleNamespace(slot_id=101), _active_requests=0, proxy=proxy)
        self.router._get_or_start = AsyncMock(return_value=[backend])
        sent = self._multipart()
        response = await self._post(body=sent)
        self.assertEqual(200, response.status)
        self.assertEqual(sent, seen["body"])
        self.assertIn(self.BOUNDARY, seen["ctype"])
        self.router._get_or_start.assert_awaited_once_with(ASR)

    async def test_transcription_model_on_other_routes_is_rejected_before_gpu_start(self):
        self.router._get_or_start = AsyncMock()
        for path in ("/v1/chat/completions", "/v1/audio/speech", "/v1/completions"):
            with self.subTest(path=path):
                response = await self._post(path=path)
                self.assertEqual(400, response.status)
        self.router._get_or_start.assert_not_awaited()

    async def test_gpu_busy_is_a_503_with_the_reason(self):
        self.router._get_or_start = AsyncMock(side_effect=GPUBusyError("no room: need 6.3 GiB"))
        response = await self._post()
        self.assertEqual(503, response.status)
        body = await response.json()
        self.assertEqual("gpu_busy", body["error"]["type"])
        self.assertIn("6.3 GiB", body["error"]["message"])

    async def test_unknown_model_in_a_multipart_upload_is_404(self):
        self.router._get_or_start = AsyncMock()
        response = await self._post(body=self._multipart(model="nope"))
        self.assertEqual(404, response.status)
        self.router._get_or_start.assert_not_awaited()


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
