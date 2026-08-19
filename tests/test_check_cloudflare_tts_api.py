from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "check_cloudflare_tts_api.py"
SPEC = importlib.util.spec_from_file_location("check_cloudflare_api", SCRIPT)
assert SPEC and SPEC.loader
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)


class CheckApiTests(unittest.TestCase):
    def test_token_from_env_prefers_cf_prefixed_names(self):
        with mock.patch.dict(
            checker.os.environ,
            {
                "CLOUDFLARE_API_TOKEN": "api-token",
                "CF_API_TOKEN": "cf-token",
            },
            clear=True,
        ):
            self.assertEqual("api-token", checker._token_from_env())

    def test_token_from_env_missing_exits(self):
        with mock.patch.dict(checker.os.environ, {}, clear=True):
            with self.assertRaises(SystemExit) as ctx:
                checker._token_from_env()
            self.assertIn("Missing token", str(ctx.exception))

    def test_is_mp3_signature_check(self):
        self.assertTrue(checker.is_mp3(b"ID3\x03\x00sample"))
        self.assertTrue(checker.is_mp3(bytes([0xFF, 0xFB, 0x40, 0x00])))
        self.assertFalse(checker.is_mp3(b"not-mp3"))

    def test_main_returns_zero_on_success(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tts.smoke.mp3"
            with mock.patch.object(
                checker, "_token_from_env", return_value="token"
            ), mock.patch.object(checker, "request_json", return_value={"success": True}), mock.patch.object(
                checker,
                "request_bytes",
                return_value=({"Content-Type": "audio/mpeg"}, b"ID3\x03\x00abc"),
            ):
                with mock.patch.object(checker.sys, "argv", [
                    "check_cloudflare_tts_api.py",
                    "--account-id",
                    "acct-1",
                    "--out",
                    str(output),
                ]):
                    self.assertEqual(0, checker.main())
            self.assertEqual(b"ID3\x03\x00abc", output.read_bytes())

    def test_main_rejects_unexpected_content_type(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "tts.smoke.mp3"
            with mock.patch.object(
                checker, "_token_from_env", return_value="token"
            ), mock.patch.object(checker, "request_json", return_value={"success": True}), mock.patch.object(
                checker,
                "request_bytes",
                return_value=({"Content-Type": "audio/wav"}, b"RIFF"),
            ):
                with mock.patch.object(
                    checker.sys,
                    "argv",
                    [
                        "check_cloudflare_tts_api.py",
                        "--account-id",
                        "acct-1",
                        "--out",
                        str(output),
                    ],
                ):
                    with self.assertRaises(SystemExit):
                        checker.main()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
