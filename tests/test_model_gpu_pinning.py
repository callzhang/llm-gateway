"""MODEL_GPU_IDS pins a model to specific GPUs, so a slot can be kept for a model that needs a whole card."""

import unittest

from model_manager import MODEL_CONFIGS, _pin_model_gpus

TTS = "qwen3-tts-1.7b-customvoice"
CHAT = "qwen3.8-27b"


class PinModelGpusTest(unittest.TestCase):
    def test_empty_spec_changes_nothing(self):
        self.assertEqual(_pin_model_gpus(MODEL_CONFIGS, ""), MODEL_CONFIGS)

    def test_pins_each_named_model(self):
        pinned = _pin_model_gpus(MODEL_CONFIGS, f"{CHAT}=0,2,3,4; {TTS}=5")
        self.assertEqual(pinned[CHAT].allowed_gpu_ids, {0, 2, 3, 4})
        self.assertEqual(pinned[TTS].allowed_gpu_ids, {5})

    def test_other_models_stay_unconstrained(self):
        pinned = _pin_model_gpus(MODEL_CONFIGS, f"{TTS}=5")
        self.assertEqual(pinned[CHAT], MODEL_CONFIGS[CHAT])

    def test_unknown_model_is_refused(self):
        with self.assertRaises(ValueError):
            _pin_model_gpus(MODEL_CONFIGS, "not-a-model=1")

    def test_model_without_gpus_is_refused(self):
        with self.assertRaises(ValueError):
            _pin_model_gpus(MODEL_CONFIGS, f"{TTS}=")


if __name__ == "__main__":
    unittest.main()
