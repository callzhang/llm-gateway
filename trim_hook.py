"""LiteLLM pre-call hook: keep input + max_tokens ≤ model context window.

Registered in config.yaml:
    litellm_settings:
      callbacks:
        - trim_hook.context_trim_hook

Two layered guarantees, in this order:
  1. **Dynamic max_tokens cap**: every request gets `max_tokens` shrunk to
     `ctx_win - input_token_estimate - SAFETY_MARGIN` (floored at MIN_OUTPUT).
     This alone is enough to prevent the "input + output > context window"
     400 in the common case where input fits comfortably.
  2. **Input trim**: when input alone is so large that even MIN_OUTPUT
     wouldn't fit, drop oldest (user, assistant) history pairs until it does.
     System prompt and the latest user turn are always kept.

If even system + last user exceeds the trim target, we pass through with the
clamped max_tokens and let vLLM return 400 — no algorithmic way out at that
point.

Token estimation uses an ASCII/non-ASCII heuristic (4 chars/token for ASCII,
1 char/token for CJK) — close enough at the context-window boundary to keep
us from either over-trimming English or under-trimming Chinese.  Exact counts
would require running the model's tokenizer, which is too heavy to do per
request.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import yaml
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("litellm.trim_hook")

# Per-model (context_window_tokens, default_max_output).  Single source of
# truth is config.yaml — the SAME file LiteLLM loads (run_litellm.sh points it
# at $SCRIPT_DIR/config.yaml, and this hook lives in that dir).  We read
# model_info.max_context_window_tokens and litellm_params.max_tokens rather
# than hardcoding them here, so ctx limits can't silently drift from what vLLM
# actually serves.  default_max_output is only used when the request itself
# carries no max_tokens; the hook caps whichever value it sees against input.
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
_DEFAULT_CTX = 32768
_DEFAULT_OUT = 4096


def _load_model_limits(path: str) -> dict[str, tuple[int, int]]:
    """Parse config.yaml into {model_name: (ctx_win, default_max_out)}.

    Missing model_info.max_context_window_tokens → skip (model falls back to
    _DEFAULT_CTX at lookup time).  Any parse failure → empty map + defaults, so
    a malformed config degrades gracefully instead of crashing the proxy."""
    try:
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("trim_hook: could not load %s (%s) — using defaults", path, e)
        return {}

    limits: dict[str, tuple[int, int]] = {}
    for entry in cfg.get("model_list") or []:
        name = entry.get("model_name")
        ctx = (entry.get("model_info") or {}).get("max_context_window_tokens")
        max_out = (entry.get("litellm_params") or {}).get("max_tokens")
        if name and ctx:
            limits[name] = (int(ctx), int(max_out) if max_out else _DEFAULT_OUT)
    return limits


_MODEL_LIMITS: dict[str, tuple[int, int]] = _load_model_limits(_CONFIG_PATH)
logger.info("trim_hook: loaded model limits from %s: %s", _CONFIG_PATH, _MODEL_LIMITS)

# Leave a small reserve for chat-template overhead (BOS/EOS, role markers,
# tokenizer rounding), so the actual prompt has room to be ≤ ctx_win.
_SAFETY_MARGIN = 512

# Minimum output budget.  If dynamic capping would push max_tokens below this,
# we trim input messages instead — preserving the model's ability to actually
# answer rather than truncating mid-token.
_MIN_OUTPUT = 1024


# Qwen3.6's chat template raises `TemplateError: No user query found in
# messages.` (→ vLLM 400) whenever the conversation has no user-role turn —
# e.g. a system/instructions-only summarization request.  OpenAI models accept
# such requests, so callers written against the OpenAI API (friday-memory's
# memory-connector among them) send them and get a hard 400.  We append a
# minimal user turn to satisfy the template; "." is enough to give the
# generation prompt something to anchor on without steering the output.
_FILLER_USER = "."


def _has_user_role(items: list[dict]) -> bool:
    return any(
        isinstance(m, dict) and m.get("role") == "user" for m in items
    )


def _estimate_tokens(text: str) -> int:
    """Rough token count.  ASCII is ~4 chars/token (English BPE), non-ASCII is
    closer to 1 char/token (CJK).  Sum both for mixed-script content."""
    if not isinstance(text, str) or not text:
        return 0
    ascii_chars = sum(1 for c in text if c.isascii())
    return ascii_chars // 4 + (len(text) - ascii_chars)


def _msg_tokens(msg: dict) -> int:
    c = msg.get("content", "")
    if isinstance(c, str):
        return _estimate_tokens(c)
    if isinstance(c, list):
        return sum(
            _estimate_tokens(p.get("text", ""))
            for p in c if isinstance(p, dict)
        )
    return 0


def _trim_messages(
    messages: list[dict],
    token_budget: int,
) -> tuple[list[dict], int]:
    """Drop oldest (user, assistant) pairs from history until total token
    estimate ≤ token_budget.  System prompt and the last non-system message
    are always retained.

    Returns (trimmed_messages, n_pairs_dropped).
    If mandatory content alone exceeds budget, returns (original, 0).
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system  = [m for m in messages if m.get("role") != "system"]
    if not non_system:
        return messages, 0

    last_msg = [non_system[-1]]
    history  = non_system[:-1]

    mandatory_tokens = sum(_msg_tokens(m) for m in system_msgs + last_msg)
    if mandatory_tokens > token_budget:
        return messages, 0

    # Group history into (user, assistant?) pairs so we drop turns atomically.
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
            i += 1   # tool / other roles: skip (rare in this pipeline)

    remaining = token_budget - mandatory_tokens
    included: list[tuple[dict, ...]] = []
    for pair in reversed(pairs):
        pair_tokens = sum(_msg_tokens(m) for m in pair)
        if pair_tokens > remaining:
            break   # this pair (and everything older) doesn't fit
        included.insert(0, pair)
        remaining -= pair_tokens

    n_dropped = len(pairs) - len(included)
    if n_dropped == 0:
        return messages, 0

    trimmed = system_msgs + [m for pair in included for m in pair] + last_msg
    return trimmed, n_dropped


class ContextTrimHook(CustomLogger):
    """Dynamic max_tokens cap + reactive message trim so input + output fits."""

    def __init__(self) -> None:
        super().__init__()

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        # Responses API path.  LiteLLM converts `input`→`messages` DOWNSTREAM of
        # this hook (inside the custom_openai provider), so here the data still
        # carries `input`/`instructions` and no `messages`.  The token-trim
        # logic below doesn't apply to this shape; the one thing we must do is
        # guarantee a user turn so the Qwen template doesn't 400.
        if call_type == "aresponses":
            return self._ensure_responses_user_turn(data)

        if call_type not in ("completion", "acompletion"):
            return None

        messages: list[dict] | None = data.get("messages")
        if not messages:
            return None

        # Same Qwen "No user query found in messages" guard for the chat path.
        injected = False
        if not _has_user_role(messages):
            messages = list(messages) + [
                {"role": "user", "content": _FILLER_USER}
            ]
            data["messages"] = messages
            injected = True
            logger.warning(
                "trim_hook: chat request for %s had no user turn (%d msgs) — "
                "appended filler user turn to satisfy the chat template",
                data.get("model"), len(messages) - 1,
            )

        # Strip provider prefix: "custom_openai/qwen3.6-35b-a3b" → "qwen3.6-35b-a3b"
        model_name = (data.get("model") or "").rsplit("/", 1)[-1]
        ctx_win, default_max_out = _MODEL_LIMITS.get(
            model_name, (_DEFAULT_CTX, _DEFAULT_OUT)
        )
        requested_max = data.get("max_tokens") or default_max_out

        input_est = sum(_msg_tokens(m) for m in messages)
        changed = False

        # Step 1: if input alone leaves less than _MIN_OUTPUT for the model to
        # speak, trim oldest history pairs until at least _MIN_OUTPUT fits.
        trim_target = ctx_win - _MIN_OUTPUT - _SAFETY_MARGIN
        if input_est > trim_target:
            trimmed, n_dropped = _trim_messages(messages, trim_target)
            if n_dropped > 0:
                new_input = sum(_msg_tokens(m) for m in trimmed)
                logger.warning(
                    "trim_hook: trimmed %d pair(s) for %s "
                    "(input_est tokens: %d → %d, msgs: %d → %d, ctx_win=%d)",
                    n_dropped, model_name,
                    input_est, new_input,
                    len(messages), len(trimmed),
                    ctx_win,
                )
                data["messages"] = trimmed
                input_est = new_input
                changed = True
            else:
                # System + last user alone is already over.  Fall through to
                # cap max_tokens to _MIN_OUTPUT and let vLLM decide.
                logger.warning(
                    "trim_hook: mandatory content (%d tokens) > trim target "
                    "(%d) for %s — passing through with capped max_tokens",
                    input_est, trim_target, model_name,
                )

        # Step 2: always cap max_tokens so input + output fits.  This is the
        # primary guarantee — even with input_est == 0 we still apply it so
        # an oversized requested_max gets clipped to the model's actual ceiling.
        #
        # NO _MIN_OUTPUT floor here on purpose: when input alone leaves <1024
        # tokens of room, forcing max_tokens up to 1024 would re-create the
        # overflow we're trying to prevent.  Instead we shrink max_tokens to
        # the actual remaining budget (possibly <1024) and just warn — the
        # model may truncate its answer, which is strictly better than 400.
        raw_safe = ctx_win - input_est - _SAFETY_MARGIN
        if raw_safe < 1:
            # Input alone exceeds (ctx_win - safety): vLLM will still 400.
            # Set max_tokens=1 so OpenAI API doesn't reject for non-positive
            # max_tokens — vLLM's own context check will produce the 400 with
            # a clear "input too large" message instead of our crashing here.
            logger.warning(
                "trim_hook: input_est %d ≥ ctx_win %d - safety %d for %s — "
                "vLLM will return 400 (no algorithmic recourse)",
                input_est, ctx_win, _SAFETY_MARGIN, model_name,
            )
            new_max = 1
        else:
            new_max = min(requested_max, raw_safe)
            if new_max < _MIN_OUTPUT:
                logger.warning(
                    "trim_hook: cramped output budget %d < soft min %d for %s "
                    "(input_est=%d, ctx_win=%d) — model may truncate",
                    new_max, _MIN_OUTPUT, model_name, input_est, ctx_win,
                )

        if new_max != data.get("max_tokens"):
            logger.info(
                "trim_hook: capped max_tokens %s → %d for %s "
                "(input_est=%d, ctx_win=%d)",
                data.get("max_tokens"), new_max, model_name, input_est, ctx_win,
            )
            data["max_tokens"] = new_max
            changed = True

        return data if (changed or injected) else None

    def _ensure_responses_user_turn(self, data: dict) -> dict | None:
        """Responses API analogue of the chat-path user-turn guard.

        `data["input"]` is either a bare string (already a user turn once
        non-empty) or a list of role-tagged items.  If no user turn is present,
        append a filler one so the downstream chat-template conversion doesn't
        400 with "No user query found in messages."  `instructions` maps to a
        system message and never counts as the user turn.
        """
        input_val = data.get("input")

        if isinstance(input_val, str):
            if input_val.strip():
                return None            # non-empty string is the user turn
            items: list[dict] = []     # empty string → treat as no user turn
        elif isinstance(input_val, list):
            if _has_user_role(input_val):
                return None
            items = list(input_val)
        else:
            items = []                 # None / unexpected shape

        items.append({"role": "user", "content": _FILLER_USER})
        data["input"] = items
        logger.warning(
            "trim_hook: responses request for %s had no user turn — appended "
            "filler user turn to satisfy the chat template",
            data.get("model"),
        )
        return data


# Module-level instance — config.yaml references this via "trim_hook.context_trim_hook"
context_trim_hook = ContextTrimHook()
