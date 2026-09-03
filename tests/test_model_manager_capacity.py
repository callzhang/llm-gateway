from __future__ import annotations

import os
import re
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


if __name__ == "__main__":
    unittest.main()
