from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from request_priority_hook import RequestPriorityHook


def key(metadata: dict | None = None) -> SimpleNamespace:
    """The part of LiteLLM's UserAPIKeyAuth the hook reads: the virtual key's metadata."""
    return SimpleNamespace(metadata=metadata or {}, key_alias="test-key")


def run(data: dict, metadata: dict | None = None, call_type: str = "aresponses") -> dict | None:
    return asyncio.run(RequestPriorityHook().async_pre_call_hook(key(metadata), None, data, call_type))


def sent(data: dict):
    return (data.get("extra_body") or {}).get("priority")


class GrantedKeyTests(unittest.TestCase):
    GRANT = {"gateway_priority": 0}

    def test_a_granted_key_runs_at_its_priority_without_sending_one(self):
        for call_type in ("aresponses", "acompletion"):
            with self.subTest(call_type=call_type):
                data = {"model": "qwen3.8-27b", "stream": True}
                self.assertIs(run(data, self.GRANT, call_type), data)
                self.assertEqual(0, sent(data))

    def test_a_granted_key_keeps_other_extra_body_fields(self):
        data = {"model": "qwen3.8-27b", "extra_body": {"thinking_token_budget": 4096}}
        run(data, self.GRANT)
        self.assertEqual({"thinking_token_budget": 4096, "priority": 0}, data["extra_body"])

    def test_a_granted_key_may_ask_for_a_lower_priority(self):
        data = {"model": "qwen3.8-27b", "extra_body": {"priority": 200}}
        run(data, {"gateway_priority": 10})
        self.assertEqual(200, sent(data))

    def test_a_granted_key_cannot_ask_above_its_grant(self):
        data = {"model": "qwen3.8-27b", "extra_body": {"priority": 0}}
        run(data, {"gateway_priority": 10})
        self.assertEqual(10, sent(data))


class UngrantedKeyTests(unittest.TestCase):
    def test_priority_zero_is_raised_to_one(self):
        for metadata in (None, {"purpose": "memory-ci-controller via tailscale"}):
            with self.subTest(metadata=metadata):
                data = {"model": "qwen3.8-27b", "extra_body": {"priority": 0}}
                self.assertIs(run(data, metadata), data)
                self.assertEqual(1, sent(data))

    def test_other_priorities_pass_unchanged(self):
        for priority in (1, 100, 200, 255):
            with self.subTest(priority=priority):
                data = {"model": "qwen3.8-27b", "extra_body": {"priority": priority}}
                self.assertIsNone(run(data))
                self.assertEqual(priority, sent(data))

    def test_no_priority_stays_absent_so_model_manager_applies_its_default(self):
        data = {"model": "qwen3.8-27b", "input": "hi"}
        self.assertIsNone(run(data))
        self.assertNotIn("extra_body", data)

    def test_a_malformed_priority_is_left_for_model_manager_to_reject(self):
        for bad in (-1, 256, 1.5, "high", True, None):
            with self.subTest(bad=bad):
                data = {"model": "qwen3.8-27b", "extra_body": {"priority": bad}}
                self.assertIsNone(run(data, {"gateway_priority": 0}))
                self.assertEqual(bad, data["extra_body"]["priority"])


class GrantValidationTests(unittest.TestCase):
    def test_an_invalid_grant_grants_nothing(self):
        for grant in ("0", -1, 256, 0.0, True, None):
            with self.subTest(grant=grant):
                data = {"model": "qwen3.8-27b", "extra_body": {"priority": 0}}
                run(data, {"gateway_priority": grant})
                self.assertEqual(1, sent(data))


class CallTypeTests(unittest.TestCase):
    def test_non_chat_calls_are_untouched(self):
        for call_type in ("aembedding", "aspeech", "atranscription", "aimage_generation"):
            with self.subTest(call_type=call_type):
                data = {"model": "qwen3-asr"}
                self.assertIsNone(run(data, {"gateway_priority": 0}, call_type))
                self.assertNotIn("extra_body", data)


if __name__ == "__main__":
    unittest.main()
