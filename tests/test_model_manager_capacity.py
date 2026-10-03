from __future__ import annotations

import asyncio
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import mock_open, patch

import model_manager


class ModelSequenceLimitConfigTests(unittest.TestCase):
    def test_qwen38_registered_limit_matches_validated_gpu4_capacity(self) -> None:
        # 8 validated by the 2026-09-03 A/B: rho=9.0 peak demand, per-request
        # decode -1.4% at batch 8, KV 31.8k tokens/seq.  See MODEL_CONFIGS.
        self.assertEqual(8, model_manager.MODEL_CONFIGS["qwen3.8-27b"].max_num_seqs)

    def test_chat_model_accepts_positive_max_num_seqs(self) -> None:
        config = model_manager.ModelConfig(
            script="run_qwen38_27b.sh",
            served_name="qwen3.8-27b",
            max_num_seqs=4,
        )

        self.assertEqual(4, config.max_num_seqs)

    def test_chat_model_rejects_invalid_max_num_seqs(self) -> None:
        for value in (None, 0, -1, True, 4.0, "4"):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    model_manager.ModelConfig(
                        script="run_qwen38_27b.sh",
                        served_name="qwen3.8-27b",
                        max_num_seqs=value,
                    )

    def test_registered_chat_models_declare_positive_max_num_seqs(self) -> None:
        for model_name, config in model_manager.MODEL_CONFIGS.items():
            if config.request_kind != "chat":
                continue
            with self.subTest(model_name=model_name):
                self.assertIs(type(config.max_num_seqs), int)
                self.assertGreater(config.max_num_seqs, 0)


class AdoptedBackendConfigTests(unittest.TestCase):
    def test_adoption_rejects_backend_with_stale_memory_utilization(self) -> None:
        cmdline = [
            "vllm",
            "serve",
            "model",
            "--served-model-name",
            "qwen3.8-27b",
            "--gpu-memory-utilization",
            "0.960",
            "--max-num-seqs",
            "4",
        ]
        config = model_manager.ModelConfig(
            script="run_qwen38_27b.sh",
            served_name="qwen3.8-27b",
            max_num_seqs=4,
        )

        with patch.object(model_manager, "_read_vllm_cmdline", return_value=cmdline):
            self.assertFalse(
                model_manager._adopted_vllm_matches_config(123, "qwen3.8-27b", config)
            )

    def test_adoption_accepts_lower_runtime_utilization_from_vram_clamp(self) -> None:
        cmdline = [
            "vllm",
            "serve",
            "model",
            "--served-model-name",
            "qwen3.8-27b",
            "--gpu-memory-utilization",
            "0.800",
            "--max-num-seqs",
            "4",
        ]
        config = model_manager.ModelConfig(
            script="run_qwen38_27b.sh",
            served_name="qwen3.8-27b",
            max_num_seqs=4,
        )

        with patch.object(model_manager, "_read_vllm_cmdline", return_value=cmdline):
            self.assertTrue(
                model_manager._adopted_vllm_matches_config(123, "qwen3.8-27b", config)
            )


class ModelSequenceLimitRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_spawn_passes_configured_max_num_seqs(self) -> None:
        backend = model_manager.GpuBackend(
            "qwen3.8-27b",
            "run_qwen38_27b.sh",
            "qwen3.8-27b",
            model_manager.GpuSlot(0, 0, 9010),
            max_num_seqs=4,
        )
        process = SimpleNamespace(pid=12345, poll=lambda: 1)

        with patch("builtins.open", mock_open()), patch.object(
            model_manager.subprocess, "Popen", return_value=process
        ) as popen_mock, patch.object(os, "killpg"):
            started = await backend._spawn_attempt_locked(util=None)

        self.assertFalse(started)
        self.assertEqual("4", popen_mock.call_args.kwargs["env"]["VLLM_MAX_NUM_SEQS"])

    def test_status_exposes_limits_independent_of_slot_state(self) -> None:
        configs = {
            "qwen3.8-27b": model_manager.ModelConfig(
                script="run_qwen38_27b.sh",
                served_name="qwen3.8-27b",
                max_num_seqs=4,
            )
        }
        router = model_manager.DynamicRouter(
            [model_manager.GpuSlot(0, 0, 9010)],
            configs,
        )

        self.assertEqual(
            {"qwen3.8-27b": {"max_num_seqs": 4}},
            router.status()["model_limits"],
        )


class ChatModelUtilCeilingTests(unittest.TestCase):
    # RTX 5090 total and the util-independent footprint (CUDA context +
    # non-torch memory) measured 2026-09-26: 30108 MiB used at util 0.90.
    TOTAL_MIB = 32607.0
    OVERHEAD_MIB = 30108.0 - 0.90 * TOTAL_MIB

    def test_chat_models_share_one_ceiling(self) -> None:
        for name, config in model_manager.MODEL_CONFIGS.items():
            if config.request_kind != "chat":
                continue
            with self.subTest(model=name):
                self.assertEqual(
                    model_manager.CHAT_GPU_MEM_UTIL_CEILING,
                    model_manager.MODEL_GPU_MEM_UTIL[name],
                )

    def test_ceiling_footprint_fits_an_empty_card(self) -> None:
        footprint = (
            model_manager.CHAT_GPU_MEM_UTIL_CEILING * self.TOTAL_MIB
            + self.OVERHEAD_MIB
        )
        self.assertLess(footprint, self.TOTAL_MIB)

    def test_chat_model_util_stays_above_min_viable_floor(self) -> None:
        for name, config in model_manager.MODEL_CONFIGS.items():
            if config.request_kind != "chat":
                continue
            with self.subTest(model=name):
                self.assertGreaterEqual(
                    model_manager.MODEL_GPU_MEM_UTIL[name],
                    model_manager.MODEL_MIN_GPU_MEM_UTIL[name],
                )


class ModelSequenceLimitLauncherTests(unittest.TestCase):
    def test_warm_cache_derives_spawn_env_instead_of_hardcoding(self) -> None:
        # The warm script must pull VLLM_MAX_NUM_SEQS / VLLM_GPU_MEM_UTIL out
        # of model_manager.py at runtime.  A literal value here would be a
        # second owner that drifts from MODEL_CONFIGS — the bug class that
        # produced the dead 0.88 on 2026-09-03.
        warm_script = (
            Path(model_manager.SCRIPT_DIR) / "scripts" / "warm_jit_cache.sh"
        ).read_text(encoding="utf-8")

        self.assertIsNone(
            re.search(r"export VLLM_MAX_NUM_SEQS=\d", warm_script),
            "warm_jit_cache.sh hardcodes VLLM_MAX_NUM_SEQS",
        )
        self.assertIsNone(
            re.search(r"export VLLM_GPU_MEM_UTIL=[\d.]", warm_script),
            "warm_jit_cache.sh hardcodes VLLM_GPU_MEM_UTIL",
        )
        self.assertIn("import model_manager", warm_script)
        self.assertIn("MODEL_GPU_MEM_UTIL", warm_script)

    def test_chat_launchers_require_model_manager_gpu_mem_util(self) -> None:
        # No :-default fallback allowed: the manager (MODEL_GPU_MEM_UTIL) is
        # the only owner of this value, so a bare-run script must fail loudly.
        for launcher_name in ("run_qwen38_27b.sh", "run_qwen36_35b_heretic.sh"):
            with self.subTest(launcher=launcher_name):
                launcher = (
                    Path(model_manager.SCRIPT_DIR) / launcher_name
                ).read_text(encoding="utf-8")
                self.assertRegex(launcher, r"\$\{VLLM_GPU_MEM_UTIL:\?")
                self.assertNotRegex(launcher, r"\$\{VLLM_GPU_MEM_UTIL:-")

    def test_chat_launchers_require_model_manager_sequence_limit(self) -> None:
        for launcher_name in ("run_qwen38_27b.sh", "run_qwen36_35b_heretic.sh"):
            with self.subTest(launcher=launcher_name):
                launcher = (
                    Path(model_manager.SCRIPT_DIR) / launcher_name
                ).read_text(encoding="utf-8")
                self.assertIn(
                    "VLLM_MAX_NUM_SEQS:?VLLM_MAX_NUM_SEQS is required",
                    launcher,
                )
                self.assertIn('--max-num-seqs "$MAX_NUM_SEQS"', launcher)


class ScaleInQuietWindowTests(unittest.TestCase):
    def test_quiet_window_outlasts_scale_out_reaction(self) -> None:
        # A replica must stay quiet longer than it would take scale-out to
        # decide it is needed again; otherwise reclaim/re-spawn can cycle.
        fastest_tier = min(secs for _, secs in model_manager.SCALE_OUT_TIERS)
        self.assertGreater(
            model_manager.REPLICA_QUIET_BEFORE_RECLAIM, fastest_tier
        )
        self.assertGreaterEqual(
            model_manager.REPLICA_QUIET_BEFORE_RECLAIM,
            model_manager.REPLICA_IDLE_TIMEOUT,
        )

    def test_primary_idle_timeout_outlasts_replica_shedding(self) -> None:
        self.assertGreater(
            model_manager.IDLE_TIMEOUT, model_manager.REPLICA_IDLE_TIMEOUT
        )


if __name__ == "__main__":
    unittest.main()


class EvalLockShutdownTests(unittest.IsolatedAsyncioTestCase):
    async def test_shutdown_waits_until_eval_lock_is_released(self) -> None:
        with tempfile.NamedTemporaryFile() as lock_file:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            task = asyncio.create_task(
                model_manager._wait_for_eval_lock_release(
                    model_manager.logging.getLogger("test"),
                    lock_path=lock_file.name,
                    poll_seconds=0.001,
                )
            )
            await asyncio.sleep(0.01)
            self.assertFalse(task.done())
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            await asyncio.wait_for(task, timeout=1.0)
