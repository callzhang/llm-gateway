"""SGLang logs its whole ServerArgs at startup, the engine key included; our plugin masks it."""
from __future__ import annotations

import dataclasses
import sys
import types
import unittest
from importlib.metadata import entry_points
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "sglang_plugins"))


@dataclasses.dataclass
class ServerArgs:
    model_path: str = "m"
    api_key: str | None = None
    admin_api_key: str | None = None


class RedactTests(unittest.TestCase):
    def test_the_keys_are_masked_wherever_the_args_are_printed(self):
        import friday_redact_keys

        module = types.ModuleType("sglang.srt.server_args")
        module.ServerArgs = ServerArgs
        with mock.patch.dict(sys.modules, {"sglang": types.ModuleType("sglang"), "sglang.srt": types.ModuleType("sglang.srt"),
                                          "sglang.srt.server_args": module}):
            friday_redact_keys.register()
        server_args = ServerArgs(api_key="secret-1", admin_api_key="secret-2")
        text = f"{server_args=}"
        self.assertNotIn("secret", text)
        self.assertIn("model_path='m'", text)
        self.assertEqual("secret-1", server_args.api_key)
        self.assertIn("api_key=None", repr(ServerArgs()))

    def test_sglang_can_discover_it_and_the_run_script_enables_it(self):
        found = {e.name: e.value for e in entry_points(group="sglang.srt.plugins")}
        self.assertEqual("friday_redact_keys:register", found.get("friday_redact_keys"))
        script = (ROOT / "run_qwen38_27b_sglang.sh").read_text()
        self.assertRegex(script, r"(?m)^export SGLANG_PLUGINS=.*\bfriday_redact_keys\b")


if __name__ == "__main__":
    unittest.main()
