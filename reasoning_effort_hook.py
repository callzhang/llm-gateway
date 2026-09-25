"""LiteLLM pre-call hook: map `reasoning_effort` onto the levels a model's chat template accepts.

Registered in config.yaml:
    litellm_settings:
      callbacks:
        - reasoning_effort_hook.reasoning_effort_hook

Why it exists: `drop_params: true` makes LiteLLM silently drop `reasoning_effort` for
`custom_openai` models, so every qwen3.8 request ran at the chat template's default (`xhigh`,
the longest thinking) whatever the client asked for. Passing the parameter through
(`allowed_openai_params`) is not enough on its own: the Qwen3.8 template raises on anything
outside `xhigh | medium | low`, so the OpenAI-style `high` that clients send would turn into 400s.

For each model that declares `model_info.reasoning_effort` in config.yaml this hook:
  - keeps a value the model supports,
  - maps an alias onto a supported level (e.g. `high` -> `xhigh`),
  - removes anything else, which falls back to the template default exactly as before.
Models without that block are left untouched.

    model_info:
      reasoning_effort:
        levels: [xhigh, medium, low]
        aliases: {high: xhigh, minimal: low}
"""
from __future__ import annotations

import logging
import os
from typing import Any

import yaml
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("litellm.reasoning_effort_hook")

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")


def _load_policies(path: str) -> dict[str, tuple[frozenset[str], dict[str, str]]]:
    """Parse config.yaml into {model_name: (levels, aliases)}; a bad config disables the hook."""
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("reasoning_effort_hook: could not load %s (%s) — passing requests through", path, e)
        return {}
    policies: dict[str, tuple[frozenset[str], dict[str, str]]] = {}
    for entry in cfg.get("model_list") or []:
        spec = (entry.get("model_info") or {}).get("reasoning_effort")
        if not isinstance(spec, dict) or not spec.get("levels"):
            continue
        levels = frozenset(str(level) for level in spec["levels"])
        aliases = {str(k): str(v) for k, v in (spec.get("aliases") or {}).items() if str(v) in levels}
        policies[str(entry.get("model_name"))] = (levels, aliases)
    return policies


class ReasoningEffortHook(CustomLogger):
    """Normalise reasoning effort per model so it reaches the template in a form it accepts."""

    def __init__(self, config_path: str = _CONFIG_PATH) -> None:
        super().__init__()
        self._policies = _load_policies(config_path)

    def _resolve(self, model: str, value: Any) -> str | None:
        levels, aliases = self._policies[model]
        effort = str(value).strip().lower()
        if effort in levels:
            return effort
        if effort in aliases:
            return aliases[effort]
        logger.warning("reasoning_effort_hook: %s does not support reasoning_effort=%r; using the template default", model, value)
        return None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        model = str(data.get("model") or "")
        if model not in self._policies:
            return None
        changed = False
        # Chat completions carry `reasoning_effort`; the Responses API carries `reasoning: {effort}`.
        if data.get("reasoning_effort") is not None:
            resolved = self._resolve(model, data["reasoning_effort"])
            if resolved != data["reasoning_effort"]:
                if resolved is None:
                    data.pop("reasoning_effort")
                else:
                    data["reasoning_effort"] = resolved
                changed = True
        reasoning = data.get("reasoning")
        if isinstance(reasoning, dict) and reasoning.get("effort") is not None:
            resolved = self._resolve(model, reasoning["effort"])
            if resolved != reasoning["effort"]:
                if resolved is None:
                    reasoning.pop("effort")
                else:
                    reasoning["effort"] = resolved
                changed = True
        return data if changed else None


# Module-level instance — config.yaml references this via "reasoning_effort_hook.reasoning_effort_hook"
reasoning_effort_hook = ReasoningEffortHook()
