"""vllm_plugins/sitecustomize.py caps xgrammar's JSON whitespace once vLLM imports xgrammar.

xgrammar is not installed in the gateway's test environment, so a stand-in `xgrammar.compiler`
records the arguments compile_json_schema receives.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

FAKE_COMPILER = textwrap.dedent(
    """
    class GrammarCompiler:
        def compile_json_schema(self, schema, *, any_whitespace=True, max_whitespace_cnt=None):
            return {"schema": schema, "any_whitespace": any_whitespace, "max_whitespace_cnt": max_whitespace_cnt}
    """
)


def load_plugin():
    spec = importlib.util.spec_from_file_location("friday_vllm_sitecustomize", ROOT / "vllm_plugins" / "sitecustomize.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WhitespaceCapTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        pkg = pathlib.Path(self.tmp.name) / "xgrammar"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("from .compiler import GrammarCompiler\n")
        (pkg / "compiler.py").write_text(FAKE_COMPILER)
        sys.path.insert(0, self.tmp.name)
        self.meta_path = list(sys.meta_path)
        self.saved = {name: sys.modules.pop(name) for name in ("xgrammar", "xgrammar.compiler") if name in sys.modules}

    def tearDown(self):
        sys.meta_path[:] = self.meta_path
        sys.path.remove(self.tmp.name)
        for name in ("xgrammar", "xgrammar.compiler"):
            sys.modules.pop(name, None)
        sys.modules.update(self.saved)
        self.tmp.cleanup()

    def test_patches_on_first_import_of_xgrammar(self):
        load_plugin().install(20)
        self.assertNotIn("xgrammar.compiler", sys.modules)

        import xgrammar

        compiler = xgrammar.GrammarCompiler()
        self.assertEqual(compiler.compile_json_schema("{}", any_whitespace=True)["max_whitespace_cnt"], 20)
        self.assertEqual(compiler.compile_json_schema("{}")["max_whitespace_cnt"], 20)

    def test_leaves_compact_and_explicit_caps_alone(self):
        load_plugin().install(20)
        import xgrammar

        compiler = xgrammar.GrammarCompiler()
        self.assertIsNone(compiler.compile_json_schema("{}", any_whitespace=False)["max_whitespace_cnt"])
        self.assertEqual(compiler.compile_json_schema("{}", max_whitespace_cnt=3)["max_whitespace_cnt"], 3)

    def test_patches_an_already_imported_xgrammar_once(self):
        import xgrammar

        plugin = load_plugin()
        plugin.install(20)
        plugin.install(20)
        self.assertEqual(xgrammar.GrammarCompiler().compile_json_schema("{}")["max_whitespace_cnt"], 20)


class LaunchScriptTest(unittest.TestCase):
    def test_27b_allows_whitespace_and_loads_the_cap(self):
        script = (ROOT / "run_qwen38_27b.sh").read_text()
        self.assertIn("""--structured-outputs-config '{"backend":"xgrammar"}'""", script)
        self.assertNotIn('"disable_any_whitespace":true', script)
        self.assertIn("/vllm_plugins${PYTHONPATH:+:$PYTHONPATH}", script)
        self.assertIn("export VLLM_JSON_MAX_WHITESPACE=${VLLM_JSON_MAX_WHITESPACE:-20}", script)


if __name__ == "__main__":
    unittest.main()
