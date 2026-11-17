"""Cap the whitespace vLLM's xgrammar backend allows between JSON tokens.

run_qwen38_27b.sh puts this directory on PYTHONPATH and sets VLLM_JSON_MAX_WHITESPACE; every Python
process of the vLLM server (API server, engine core, workers) then imports this file at startup.

Why: vLLM offers only two whitespace modes for JSON-schema output, and both fail on qwen3.8-27b.
- disable_any_whitespace (compact) forbids the "\\n" the model puts 44-99% of its probability on
  after `"entities": [`; the masked choice lands on `]` often enough that memory extractions with
  thinking off came back empty (CN set, 2026-11-17: 16 of 48).
- any whitespace lets a model that closed a string early with a bare `"` pad whitespace up to the
  output ceiling (CN memory judges on SGLang: 16,000 tokens, about 575 s each, 2026-11-17).
xgrammar's max_whitespace_cnt keeps newlines and indentation but bounds each run, and vLLM 0.29
does not expose it, so this wraps GrammarCompiler.compile_json_schema when xgrammar is imported.
The hook is lazy: processes that never import xgrammar (torch compile workers) do not pay for it.
"""
import importlib.abc
import os
import sys

_MODULE = "xgrammar.compiler"


def patch(module, cap):
    """Make compile_json_schema default max_whitespace_cnt to `cap` whenever whitespace is allowed."""
    compiler = module.GrammarCompiler
    original = compiler.compile_json_schema
    if hasattr(original, "friday_whitespace_cap"):
        return

    def compile_json_schema(self, schema, **kwargs):
        if kwargs.get("any_whitespace", True) and kwargs.get("max_whitespace_cnt") is None:
            kwargs["max_whitespace_cnt"] = cap
        return original(self, schema, **kwargs)

    compile_json_schema.friday_whitespace_cap = cap
    compile_json_schema.__doc__ = original.__doc__
    compiler.compile_json_schema = compile_json_schema
    print(f"friday: xgrammar JSON whitespace capped at {cap} characters per run", file=sys.stderr)


class _PatchOnImport(importlib.abc.MetaPathFinder):
    def __init__(self, cap):
        self.cap = cap

    def find_spec(self, name, path, target=None):
        if name != _MODULE:
            return None
        for finder in sys.meta_path:
            if finder is self or not hasattr(finder, "find_spec"):
                continue
            spec = finder.find_spec(name, path, target)
            if spec is not None:
                break
        else:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            patch(module, self.cap)

        spec.loader.exec_module = exec_and_patch
        return spec


def install(cap):
    if _MODULE in sys.modules:
        patch(sys.modules[_MODULE], cap)
    else:
        sys.meta_path.insert(0, _PatchOnImport(cap))


# run_qwen38_27b.sh validates the value: an exception here would only be printed, and vLLM would
# start uncapped.
_cap = os.environ.get("VLLM_JSON_MAX_WHITESPACE")
if _cap:
    install(int(_cap))
