from __future__ import annotations

import unittest

from tts_access_gateway.policy import (
    MODEL,
    PolicyDenied,
    model_listing,
    validate_route,
    validate_speech_payload,
)


class RoutePolicyTests(unittest.TestCase):
    def test_only_tts_routes_are_allowed(self):
        validate_route("POST", "/v1/audio/speech")
        validate_route("GET", "/v1/models")

        for method, path in (
            ("POST", "/v1/chat/completions"),
            ("POST", "/key/generate"),
            ("GET", "/v1/audio/speech"),
            ("POST", "/v1/audio/speech/"),
        ):
            with self.subTest(method=method, path=path):
                with self.assertRaises(PolicyDenied):
                    validate_route(method, path)

    def test_models_listing_contains_only_tts_model(self):
        listing = model_listing()
        self.assertEqual("list", listing["object"])
        self.assertEqual([MODEL], [item["id"] for item in listing["data"]])


class SpeechPayloadPolicyTests(unittest.TestCase):
    def test_accepts_instruction_and_defaults_mp3(self):
        normalized = validate_speech_payload(
            {
                "model": MODEL,
                "input": " hello ",
                "voice": "Vivian",
                "instructions": " warm ",
            }
        )
        self.assertEqual(" hello ", normalized["input"])
        self.assertEqual(" warm ", normalized["instructions"])
        self.assertEqual("mp3", normalized["response_format"])

    def test_payload_is_tts_model_voice_length_and_mp3_only(self):
        cases = (
            {"model": "qwen3.6-27b"},
            {"voice": "clone-me"},
            {"input": "x" * 3001},
            {"input": "   "},
            {"response_format": "wav"},
        )
        for mutation in cases:
            payload = {
                "model": MODEL,
                "input": "hello",
                "voice": "Vivian",
                "response_format": "mp3",
                **mutation,
            }
            with self.subTest(mutation=mutation):
                with self.assertRaises(PolicyDenied):
                    validate_speech_payload(payload)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaisesRegex(PolicyDenied, "unsupported fields"):
            validate_speech_payload(
                {
                    "model": MODEL,
                    "input": "hello",
                    "voice": "Vivian",
                    "response_format": "mp3",
                    "speaker_wav": "secret.wav",
                }
            )


if __name__ == "__main__":
    unittest.main()
