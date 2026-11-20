"""GPU_NEIGHBOUR_RESERVE_MIB: VRAM on one GPU that vLLM never plans into.

The embedding-provider needs a fixed slice of one card to scale up from CPU.
Chat models otherwise take everything free at spawn (util 0.95), which left it
0.9 GiB and an OOM on every scale-up attempt (2026-11-17).  The reserve holds
whether or not the neighbour is on the GPU right now, so it is counted against
the card's total, not its free memory.  That total is CUDA's (vLLM multiplies util by it):
nvidia-smi's memory.total minus what the driver and firmware reserve.
"""

import unittest
from unittest.mock import patch

import model_manager
from model_manager import GPUBusyError, GpuBackend, GpuSlot

CHAT = "qwen3.8-27b"
ASR = "qwen3-asr-0.6b"
TOTAL_MIB = 32607.0
# nvidia-smi memory.reserved on gpu4's GPU 1: the driver and firmware hold it, no process can
# allocate it, and CUDA (so vLLM's util x total) leaves it out of the card's total.
DRIVER_RESERVED_MIB = 458.0
CUDA_TOTAL_MIB = TOTAL_MIB - DRIVER_RESERVED_MIB
RESERVE_MIB = 4096.0


def _footprint(util):
    """nvidia-smi memory of a vLLM process spawned at util."""
    return util * CUDA_TOTAL_MIB + model_manager.VLLM_PROCESS_OVERHEAD_MIB


def _util(gpu_id, free, *, vllm_used=0.0, reserve=None, model=CHAT, coresident=False,
          driver_reserved=DRIVER_RESERVED_MIB):
    backend = GpuBackend(model, "run.sh", model, GpuSlot(0, gpu_id, 9000 + gpu_id * 10), coresident=coresident)
    reserve = {1: RESERVE_MIB} if reserve is None else reserve
    with patch.dict(model_manager.GPU_NEIGHBOUR_RESERVE_MIB, reserve, clear=True), \
            patch.object(model_manager, "_gpu_total_mib", return_value=TOTAL_MIB), \
            patch.object(model_manager, "_gpu_driver_reserved_mib", return_value=driver_reserved), \
            patch.object(model_manager, "_gpu_free_mib", return_value=free), \
            patch.object(model_manager, "_gpu_vllm_used_mib", return_value=vllm_used):
        return backend._gpu_mem_util_for_spawn()


class ParseReserveTests(unittest.TestCase):
    def test_parses_gpu_to_mib_pairs(self):
        self.assertEqual({1: 4096.0}, model_manager._parse_gpu_reserve("1=4096"))
        self.assertEqual({0: 512.0, 1: 4096.0}, model_manager._parse_gpu_reserve(" 0=512 ; 1=4096 "))
        self.assertEqual({}, model_manager._parse_gpu_reserve(""))

    def test_rejects_malformed_entries(self):
        for spec in ("1", "1=", "x=4096", "1=-5"):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                model_manager._parse_gpu_reserve(spec)


class ReservedSpawnUtilTests(unittest.TestCase):
    def test_chat_model_on_the_reserved_gpu_leaves_the_slice_free(self):
        # The slice is what nvidia-smi reports free beside the chat model.  2026-11-18 on gpu4 the
        # reserve was counted against memory.total, so the driver's 458 MiB came out of the slice:
        # util 0.851 left 3,657 MiB free, 93 MiB above the embedder's 3,564 MiB peak.
        util = _util(1, free=CUDA_TOTAL_MIB)
        free_beside_it = TOTAL_MIB - DRIVER_RESERVED_MIB - _footprint(util)
        self.assertGreaterEqual(free_beside_it, RESERVE_MIB)
        self.assertEqual(0.836, util)

    def test_unreadable_driver_reserve_refuses_the_reserved_gpu(self):
        with self.assertRaisesRegex(RuntimeError, "driver"):
            _util(1, free=CUDA_TOTAL_MIB, driver_reserved=None)

    def test_other_gpus_keep_the_ceiling(self):
        self.assertEqual(model_manager.CHAT_GPU_MEM_UTIL_CEILING, _util(0, free=32000.0))

    def test_neighbour_already_holding_its_slice_is_not_counted_twice(self):
        # The embedding-provider holds 3 GiB of its 4 GiB slice: free VRAM already excludes it,
        # and the reserve still caps the chat model at the same util as on an empty card.
        self.assertEqual(0.836, _util(1, free=CUDA_TOTAL_MIB - 3072.0))

    def test_reserve_counts_vllm_processes_already_on_the_gpu(self):
        # 1 GiB of another vLLM process is on the GPU: the primary gets that much less.
        util = _util(1, free=CUDA_TOTAL_MIB - 1024.0, vllm_used=1024.0)
        self.assertGreaterEqual(CUDA_TOTAL_MIB - _footprint(util) - 1024.0, RESERVE_MIB)
        self.assertEqual(0.804, util)

    def test_floor_is_never_clamped_up_into_the_reserve(self):
        # 8 GiB reserved leaves less than the 27B's 24.83 GiB floor: refuse rather than eat the slice.
        with self.assertRaisesRegex(RuntimeError, "reserve"):
            _util(1, free=32000.0, reserve={1: 8192.0})

    def test_coresident_model_stays_out_of_the_reserve(self):
        # Primary 27B at util 0.836 already fills the card up to the reserve.
        primary = _footprint(0.836)
        with self.assertRaisesRegex(RuntimeError, "reserve"):
            _util(1, free=CUDA_TOTAL_MIB - primary, vllm_used=primary, model=ASR, coresident=True)


class ReservedCoresidentPlacementTests(unittest.TestCase):
    def test_coresident_placement_skips_a_gpu_whose_room_is_reserved(self):
        router = model_manager.DynamicRouter(
            [GpuSlot(0, 0, 9000), GpuSlot(1, 1, 9010)], model_manager.MODEL_CONFIGS,
        )
        primary = _footprint(0.836)
        free = {0: 1500.0, 1: CUDA_TOTAL_MIB - primary}
        used = {0: CUDA_TOTAL_MIB - 1500.0, 1: primary}
        with patch.dict(model_manager.GPU_NEIGHBOUR_RESERVE_MIB, {1: RESERVE_MIB}, clear=True), \
                patch.object(model_manager, "_gpu_total_mib", return_value=TOTAL_MIB), \
                patch.object(model_manager, "_gpu_driver_reserved_mib", return_value=DRIVER_RESERVED_MIB), \
                patch.object(model_manager, "_gpu_free_mib", side_effect=free.get), \
                patch.object(model_manager, "_gpu_vllm_used_mib", side_effect=used.get):
            with self.assertRaisesRegex(GPUBusyError, "reserved"):
                router._pick_coresident_gpu(ASR)


if __name__ == "__main__":
    unittest.main()
