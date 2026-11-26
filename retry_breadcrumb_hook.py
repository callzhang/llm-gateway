"""LiteLLM callback: a retry/fallback breadcrumb describes only its own request, and no content.

Registered in config.yaml and config.cn.yaml:
    litellm_settings:
      callbacks:
        - retry_breadcrumb_hook.retry_breadcrumb_hook

LiteLLM 1.99.0's Router.log_retry appends each failed attempt's whole kwargs (Responses `input`,
and `proxy_server_request.body` with every message and base64 image) to `self.previous_models`,
one list shared by every request in the process, and hangs that list on the current request's
metadata.  LiteLLM_SpendLogs.proxy_server_request therefore stored other callers' requests, also
across virtual keys.  `num_retries: 0` does not prevent it: the order 1 -> order 2 overflow
fallback calls log_retry too.

No hook runs between log_retry and every spend-log write, so loading this module replaces
log_retry itself.  The list now lives in the request's own metadata and holds a short record per
failed attempt.  Fixed upstream in 1.101.0 (BerriAI/litellm PR 39491): delete this file after
upgrading past it (tests/test_retry_breadcrumb_hook.py says when).
"""
from __future__ import annotations

from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from litellm.router import Router

MAX_BREADCRUMBS = 4
# An engine's error text can quote the request, so keep only enough to say what failed.
MAX_EXCEPTION_CHARS = 500


def log_retry(self: Router, kwargs: dict, e: Exception) -> dict:
    name = "litellm_metadata" if "litellm_metadata" in kwargs else "metadata"
    metadata = kwargs.get(name)
    if not isinstance(metadata, dict):
        metadata = kwargs[name] = {}
    model_info = metadata.get("model_info")
    record = {
        "model_group": kwargs.get("model"),
        "deployment_id": model_info.get("id") if isinstance(model_info, dict) else None,
        "exception_type": type(e).__name__,
        "exception_string": str(e)[:MAX_EXCEPTION_CHARS],
    }
    earlier = metadata.get("previous_models")
    earlier = list(earlier) if isinstance(earlier, list) else []
    # A new list each time: whoever already holds the old one must not see it change.
    metadata["previous_models"] = [*earlier[-(MAX_BREADCRUMBS - 1):], record]
    return kwargs


Router.log_retry = log_retry


class RetryBreadcrumbHook(CustomLogger):
    """Nothing to hook: config.yaml needs an instance to load the module by."""


# Module-level instance — config.yaml references this via "retry_breadcrumb_hook.retry_breadcrumb_hook"
retry_breadcrumb_hook = RetryBreadcrumbHook()
