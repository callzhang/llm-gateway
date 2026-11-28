"""The tracked defaults must start the pinned LiteLLM, even without the overrides in gateway.env.

gateway.env is not tracked.  Until 2026-11-28 only its GATEWAY_PY_BIN line selected 1.108.19: without
it gpu4 would have started 1.99.0 without complaint, and 1.99.0 with these configs (no
retry_breadcrumb_hook) writes other callers' requests into the spend logs.
"""
import glob
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(*parts: str) -> str:
    with open(os.path.join(REPO, *parts)) as f:
        return f.read()


def pinned_version() -> str:
    [path] = glob.glob(os.path.join(REPO, "docs", "litellm-*-requirements.txt"))
    return re.search(r"^litellm==(\S+)$", read(path), re.M).group(1)


def default_env() -> str:
    return re.search(r"^GATEWAY_PY_BIN=\$\{GATEWAY_PY_BIN:-(\S+)/bin\}$", read("run_litellm.sh"), re.M).group(1)


def test_run_litellm_defaults_to_the_env_of_the_pinned_version():
    assert default_env().endswith(f"/litellm-{pinned_version()}"), default_env()


def test_the_unit_takes_prisma_engines_from_that_env():
    cache = re.search(r"^Environment=PRISMA_BINARY_CACHE_DIR=(\S+)$", read("systemd", "llm-litellm.service"), re.M).group(1)
    assert cache == f"{default_env()}/share/prisma-python"
