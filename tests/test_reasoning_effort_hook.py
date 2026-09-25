from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import yaml

from reasoning_effort_hook import ReasoningEffortHook

POLICY_CONFIG = {
    "model_list": [
        {
            "model_name": "qwen3.8-27b",
            "litellm_params": {"model": "custom_openai/qwen3.8-27b"},
            "model_info": {"reasoning_effort": {"levels": ["xhigh", "medium", "low"], "aliases": {"high": "xhigh", "minimal": "low"}}},
        },
        {"model_name": "other-model", "litellm_params": {"model": "custom_openai/other"}},
    ]
}


def run(hook: ReasoningEffortHook, data: dict, call_type: str = "acompletion") -> dict | None:
    return asyncio.run(hook.async_pre_call_hook(None, None, data, call_type))


class ReasoningEffortHookTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        path = Path(self._tmp.name) / "config.yaml"
        path.write_text(yaml.safe_dump(POLICY_CONFIG), encoding="utf-8")
        self.hook = ReasoningEffortHook(config_path=str(path))

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_supported_levels_pass_and_aliases_map(self) -> None:
        for sent, expected in (("medium", "medium"), ("low", "low"), ("xhigh", "xhigh"), ("HIGH", "xhigh"), ("high", "xhigh"), ("minimal", "low")):
            with self.subTest(sent=sent):
                data = {"model": "qwen3.8-27b", "reasoning_effort": sent}
                run(self.hook, data)
                self.assertEqual(data["reasoning_effort"], expected)

    def test_unsupported_value_is_removed_so_the_template_default_applies(self) -> None:
        data = {"model": "qwen3.8-27b", "reasoning_effort": "max"}

        self.assertIs(run(self.hook, data), data)
        self.assertNotIn("reasoning_effort", data)

    def test_unchanged_request_returns_none(self) -> None:
        self.assertIsNone(run(self.hook, {"model": "qwen3.8-27b", "reasoning_effort": "medium"}))
        self.assertIsNone(run(self.hook, {"model": "qwen3.8-27b"}))

    def test_models_without_a_policy_are_untouched(self) -> None:
        data = {"model": "other-model", "reasoning_effort": "high"}

        self.assertIsNone(run(self.hook, data))
        self.assertEqual(data["reasoning_effort"], "high")

    def test_responses_api_effort_is_normalised(self) -> None:
        data = {"model": "qwen3.8-27b", "reasoning": {"effort": "high", "summary": "auto"}}

        run(self.hook, data, call_type="aresponses")

        self.assertEqual(data["reasoning"], {"effort": "xhigh", "summary": "auto"})

    def test_unreadable_config_disables_the_hook(self) -> None:
        hook = ReasoningEffortHook(config_path=str(Path(self._tmp.name) / "missing.yaml"))
        data = {"model": "qwen3.8-27b", "reasoning_effort": "high"}

        self.assertIsNone(run(hook, data))
        self.assertEqual(data["reasoning_effort"], "high")

    def test_shipped_config_declares_the_qwen38_policy(self) -> None:
        data = {"model": "qwen3.8-27b", "reasoning_effort": "high"}

        run(ReasoningEffortHook(), data)

        self.assertEqual(data["reasoning_effort"], "xhigh")


if __name__ == "__main__":
    unittest.main()
