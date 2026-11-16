"""The qwen3.8 thinking-budget processor closes thinking once, also when output_ids lags a step.

SGLang is not installed in the gateway's test environment, so its base class is stubbed; the
processor only needs `CustomLogitProcessor` and a logits array.
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest

import numpy as np

START, END, NL = 248068, 248069, 198
VOCAB = 248100


def load_processor():
    base = types.ModuleType("sglang.srt.sampling.custom_logit_processor")
    base.CustomLogitProcessor = type("CustomLogitProcessor", (), {})
    for name in ("sglang", "sglang.srt", "sglang.srt.sampling"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["sglang.srt.sampling.custom_logit_processor"] = base
    sys.path.insert(0, "sglang_plugins")
    try:
        return importlib.reload(importlib.import_module("friday_thinking_budget")).Qwen38ThinkingBudgetLogitProcessor
    finally:
        sys.path.remove("sglang_plugins")


class Req:
    def __init__(self, prompt):
        self.origin_input_ids = prompt
        self.output_ids: list[int] = []


def decode(processor, budget, steps, lag):
    """Greedy-decode `steps` tokens of a model that would think forever (token 7) and then close thinking itself;
    the processor sees output_ids `lag` steps late, as under SGLang's overlap scheduler."""
    req = Req([1, START, NL])
    params = {"thinking_budget": budget, "__req__": req}
    produced: list[int] = []
    for _ in range(steps):
        visible = produced[:len(produced) - lag] if lag else list(produced)
        req.output_ids = visible
        logits = np.zeros((1, VOCAB))
        # The model's own preference: keep thinking, but right after a forced </think> it closes thinking again.
        if produced and produced[-1] == END:
            logits[0, END] = 10.0
        else:
            logits[0, 7] = 5.0
        logits = processor()(logits, [params])
        produced.append(int(np.argmax(logits[0])))
    return produced


class ProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Processor = load_processor()

    def test_thinking_is_closed_once_at_the_budget(self):
        for lag in (0, 1):
            out = decode(self.Processor, budget=10, steps=20, lag=lag)
            self.assertEqual(out.count(END), 1, (lag, out))
            self.assertEqual(out[out.index(END) - 1], NL, (lag, out))
            self.assertLessEqual(out.index(END), 12, (lag, out))

    def test_without_a_budget_nothing_is_forced(self):
        req = Req([1, START, NL])
        logits = np.zeros((1, VOCAB)); logits[0, 7] = 5.0
        out = self.Processor()(logits.copy(), [{"__req__": req}])
        self.assertTrue(np.array_equal(out, logits))

    def test_qwen38_think_token_ids(self):
        self.assertEqual((self.Processor.THINKING_START_TOKEN_ID, self.Processor.THINKING_END_TOKEN_ID,
                          self.Processor.NEW_LINE_TOKEN_ID), (START, END, NL))


if __name__ == "__main__":
    unittest.main()
