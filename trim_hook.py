"""LiteLLM pre-call hook: trim messages to fit within the model's context window.

Registered in config.yaml:
    litellm_settings:
      callbacks:
        - trim_hook.ContextTrimHook

The hook fires before every chat/completions call.  When the estimated token
count (1 char ≈ 1 token — conservative for Chinese/Japanese) exceeds the
budget, it drops the oldest (user, assistant) conversation pairs until the
messages fit.  The system prompt and the current user turn are always kept.

If even the mandatory content alone exceeds the budget the messages are passed
through unchanged and vLLM will return a 400 — nothing else can be done at that
point.
"""
from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("litellm.trim_hook")

# (context_window_tokens, max_output_tokens) — keep in sync with config.yaml
# max_tokens and vLLM --max-model-len.
_MODEL_LIMITS: dict[str, tuple[int, int]] = {
    "qwen3.6-35b-a3b": (122880, 32000),
    "qwen3.6-27b":     (65536,  16384),
}
_DEFAULT_CTX = 32768
_DEFAULT_OUT = 4096
_SAFETY_MARGIN = 512


def _char_count(msg: dict) -> int:
    c = msg.get("content", "")
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        return sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    return 0


def _trim_messages(
    messages: list[dict],
    budget: int,
) -> tuple[list[dict], int]:
    """Trim oldest history pairs to fit within budget chars.

    Returns (trimmed_messages, n_pairs_dropped).
    If mandatory content alone exceeds budget, returns (original, 0).
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system  = [m for m in messages if m.get("role") != "system"]

    if not non_system:
        return messages, 0

    last_msg = [non_system[-1]]   # current turn — always kept
    history  = non_system[:-1]    # eligible for trimming

    mandatory_chars = sum(_char_count(m) for m in system_msgs + last_msg)
    if mandatory_chars > budget:
        return messages, 0  # caller will let vLLM reject it

    # Group history into (user [, assistant]) pairs
    pairs: list[tuple[dict, ...]] = []
    i = 0
    while i < len(history):
        role = history[i].get("role")
        if role == "user":
            if i + 1 < len(history) and history[i + 1].get("role") == "assistant":
                pairs.append((history[i], history[i + 1]))
                i += 2
            else:
                pairs.append((history[i],))
                i += 1
        elif role == "assistant":
            pairs.append((history[i],))
            i += 1
        else:
            i += 1  # tool / other roles: skip (rare in this pipeline)

    remaining = budget - mandatory_chars
    included: list[tuple[dict, ...]] = []
    for pair in reversed(pairs):
        pair_chars = sum(_char_count(m) for m in pair)
        if pair_chars > remaining:
            break   # this pair doesn't fit; drop it and everything older
        included.insert(0, pair)
        remaining -= pair_chars

    n_dropped = len(pairs) - len(included)
    if n_dropped == 0:
        return messages, 0

    trimmed = system_msgs + [m for pair in included for m in pair] + last_msg
    return trimmed, n_dropped


class ContextTrimHook(CustomLogger):
    """Trim over-long chat histories before they reach vLLM."""

    def __init__(self) -> None:
        super().__init__()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        if call_type not in ("completion", "acompletion"):
            return None

        messages: list[dict] | None = data.get("messages")
        if not messages:
            return None

        # Strip provider prefix so "custom_openai/qwen3.6-35b-a3b" → "qwen3.6-35b-a3b"
        model_name = (data.get("model") or "").rsplit("/", 1)[-1]
        ctx_win, max_out = _MODEL_LIMITS.get(model_name, (_DEFAULT_CTX, _DEFAULT_OUT))

        # Budget: leave headroom for output tokens and overhead.
        # Floor at half the context so we don't trim absurdly when max_out is large.
        budget = max(
            min(int(ctx_win * 0.7), ctx_win - max_out - _SAFETY_MARGIN),
            ctx_win // 2,
        )

        total_chars = sum(_char_count(m) for m in messages)
        if total_chars <= budget:
            return None  # nothing to do

        trimmed, n_dropped = _trim_messages(messages, budget)
        if n_dropped == 0:
            # Mandatory content alone is too large — pass through, let vLLM fail
            logger.warning(
                "trim_hook: mandatory content exceeds budget for model=%s "
                "(chars=%d, budget=%d) — passing through",
                model_name, total_chars, budget,
            )
            return None

        trimmed_chars = sum(_char_count(m) for m in trimmed)
        logger.warning(
            "trim_hook: dropped %d pair(s) for model=%s "
            "(chars: %d → %d, budget=%d, msgs: %d → %d)",
            n_dropped, model_name,
            total_chars, trimmed_chars, budget,
            len(messages), len(trimmed),
        )
        data["messages"] = trimmed
        return data


# Module-level instance — config.yaml references this via "trim_hook.context_trim_hook"
context_trim_hook = ContextTrimHook()
