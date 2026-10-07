"""LiteLLM pre-call hook: keep input + output within the model's context window.

Registered in config.yaml:
    litellm_settings:
      callbacks:
        - trim_hook.context_trim_hook

Applies to both /v1/chat/completions and /v1/responses.  vLLM rejects a request
whose prompt + max output exceeds --max-model-len with a 400, and LiteLLM
passes that through unchanged, so the gateway has to make the request fit.

Policy, with R = context_window - input_tokens - SAFETY_MARGIN (room left for
the answer) and F = requested_output / OUTPUT_TOLERANCE (the least room we
accept without touching the input):

  * R >= requested_output            nothing to do.
  * F <= R < requested_output        shrink the output cap to R.  The model
                                     stops at the cap with finish_reason
                                     "length" (Responses: status "incomplete"),
                                     which is a normal response, not an error.
  * R < F                            shrink the input, least lossy step first,
                                     recounting after each and stopping as soon
                                     as R >= F:
                                       1. drop thinking from turns before the
                                          latest user message (Qwen's template
                                          keeps old reasoning_content, so it is
                                          billed in full for no benefit)
                                       2. replace messages that are >= 98%
                                          similar to a LATER message with a
                                          one-line stub (the latest copy is the
                                          current state; messages stay so roles
                                          and tool pairing are intact)
                                       3. cap tool-call arguments (string values
                                          inside the JSON) and tool results to
                                          model_info.tool_cap_tokens, head+tail
                                       4. drop the oldest turns.  A turn starts
                                          at a user message and runs to the next
                                          one, so tool calls stay paired with
                                          their results.
                                     Steps 1-3 only touch turns before the latest
                                     user message; system/developer messages,
                                     `instructions` and the latest turn are
                                     never modified or dropped.  Then the output
                                     cap shrinks to what is left.
  * F unreachable                    even with only the latest turn left R < F.
                                     Turns are then only dropped when R is below
                                     MIN_OUTPUT (an answer that short is useless,
                                     or the request would 400 outright), just far
                                     enough to reach MIN_OUTPUT.  Otherwise they
                                     are kept and only the cap shrinks.
  * R < 1 after all that             the mandatory content alone overflows; the
                                     request is left untouched so vLLM answers
                                     with its own explicit 400.

Requests that already fit are never modified, which keeps vLLM's prefix cache
valid.  Steps 1 and 3 are per-message and deterministic; 2 and 4 are not
prefix-stable, which only costs cache hits on conversations that overflow.

Input tokens are counted exactly: the model's own tokenizer and chat template
run on the same messages (and tool definitions) vLLM will render.  Image parts
only contribute their template placeholder, so a request heavy on images can
still undercount by up to SAFETY_MARGIN-scale amounts per image.  For
/v1/responses with previous_response_id, the stored history LiteLLM prepends
downstream is counted too, but it cannot be trimmed from here.

Models without model_info.max_input_tokens (TTS, ASR) are not touched.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import threading
from typing import Any, Callable, NamedTuple

import yaml
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("litellm.trim_hook")

# Single source of truth is config.yaml — the SAME file LiteLLM loads (run_litellm.sh
# points it at $SCRIPT_DIR/config.yaml and this hook lives in that dir).  Per model:
#   model_info.max_input_tokens   context window (== vLLM --max-model-len); LiteLLM
#                                 also registers it into litellm.model_cost
#   litellm_params.max_tokens     output cap used when the request carries none
#   model_info.tokenizer(+_revision)  HF repo of the tokenizer vLLM serves with
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
_DEFAULT_OUT = 4096

# Reserve for what the tokenizer cannot see (image tokens expand inside vLLM).
_SAFETY_MARGIN = 512

# Below this many output tokens an answer is not worth keeping history for.
_MIN_OUTPUT = 1024

# A shorter answer is accepted without trimming input when the cap loses at
# most 1 - 1/OUTPUT_TOLERANCE (~9%) of what the caller asked for.
_OUTPUT_TOLERANCE = 1.1

# Qwen3.6's chat template raises `TemplateError: No user query found in
# messages.` (→ vLLM 400) whenever the conversation has no user-role turn —
# e.g. a system/instructions-only summarization request.  OpenAI models accept
# such requests, so callers written against the OpenAI API (friday-memory's
# memory-connector among them) send them and get a hard 400.  We append a
# minimal user turn to satisfy the template; "." is enough to give the
# generation prompt something to anchor on without steering the output.
_FILLER_USER = "."

# Near-duplicate removal (step 2).  Similarity is Jaccard over word 5-grams, so
# it measures how much text two messages share, not what they mean.  Messages
# shorter than _DEDUP_MIN_CHARS are never worth a stub.
_DEDUP_SIMILARITY = 0.98
_DEDUP_MIN_CHARS = 400
_SHINGLE_WORDS = 5
_DUPLICATE_STUB = "[omitted: near-duplicate of a later message]"

# Tool arguments/results cap (step 3) when model_info.tool_cap_tokens is absent.
_DEFAULT_TOOL_CAP_TOKENS = 2000

_THINK_END = "</think>"
_FIXED_ROLES = ("system", "developer")
_CHAT_OUTPUT_KEYS = ("max_tokens", "max_completion_tokens")
_RESPONSES_OUTPUT_KEYS = ("max_output_tokens",)


class ModelLimits(NamedTuple):   # not a dataclass: LiteLLM loads this file without registering it in sys.modules
    context_window: int
    default_output: int
    tokenizer: Any
    tool_cap_tokens: int


class _Counter:
    """Exact prompt-token counting with the model's tokenizer + chat template."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._lock = threading.Lock()   # HF fast tokenizers are not re-entrant

    def count(self, messages: list[dict], tools: list[dict] | None) -> int:
        with self._lock:
            encoded = self._tokenizer.apply_chat_template(
                _template_messages(messages),
                tools=tools or None,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
            )
        return len(encoded["input_ids"])

    def encode(self, text: str) -> list[int]:
        with self._lock:
            return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        with self._lock:
            return self._tokenizer.decode(ids)


def _as_dict(msg: Any) -> dict:
    return msg if isinstance(msg, dict) else msg.model_dump(exclude_none=True)


def _template_messages(messages: list[dict]) -> list[dict]:
    """Messages as vLLM hands them to the chat template: tool-call arguments
    arrive as JSON strings on the wire but the template iterates them as dicts."""
    out = []
    for raw in messages:
        msg = _as_dict(raw)
        if msg.get("tool_calls"):
            calls = []
            for call in msg["tool_calls"]:
                call = _as_dict(call)
                fn = dict(call.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"] or "{}")
                    except json.JSONDecodeError:
                        pass   # vLLM rejects this request itself; keep counting instead of 500-ing here
                calls.append({**call, "function": fn})
            msg = {**msg, "tool_calls": calls}
        out.append(msg)
    return out


def _load_tokenizer(repo: str, revision: str | None) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(repo, revision=revision, local_files_only=True)


def _load_model_limits(
    path: str, tokenizer_loader: Callable[[str, str | None], Any]
) -> dict[str, ModelLimits]:
    """Parse config.yaml into {model_name: ModelLimits}.  A model that declares a
    context window must also name its tokenizer; a missing or unloadable
    tokenizer is a startup error, not a silent downgrade to guessing."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    limits: dict[str, ModelLimits] = {}
    for entry in cfg.get("model_list") or []:
        name = entry.get("model_name")
        info = entry.get("model_info") or {}
        ctx = info.get("max_input_tokens")
        if not (name and ctx):
            continue
        repo = info.get("tokenizer")
        if not repo:
            raise ValueError(f"trim_hook: model {name!r} declares max_input_tokens but no model_info.tokenizer")
        max_out = (entry.get("litellm_params") or {}).get("max_tokens")
        limits[name] = ModelLimits(
            context_window=int(ctx),
            default_output=int(max_out) if max_out else _DEFAULT_OUT,
            tokenizer=_Counter(tokenizer_loader(repo, info.get("tokenizer_revision"))),
            tool_cap_tokens=int(info.get("tool_cap_tokens") or _DEFAULT_TOOL_CAP_TOKENS),
        )
    return limits


def _has_user_role(items: list) -> bool:
    return any(isinstance(m, dict) and m.get("role") == "user" for m in items)


def _split_turns(items: list, role_of: Callable[[Any], str | None]) -> list[list[int]]:
    """Indices of `items` grouped into droppable turns.  A turn opens at each
    user message; everything up to the next one (assistant replies, tool calls
    and their outputs) belongs to it.  System/developer items are never in a
    turn.  Items before the first user message form their own leading turn."""
    turns: list[list[int]] = []
    for i, item in enumerate(items):
        role = role_of(item)
        if role in _FIXED_ROLES:
            continue
        if role == "user" or not turns:
            turns.append([i])
        else:
            turns[-1].append(i)
    return turns


def _drop_turns(items: list, turns: list[list[int]], n: int) -> list:
    dropped = {i for turn in turns[:n] for i in turn}
    return [item for i, item in enumerate(items) if i not in dropped]


def _fit_input(
    items: list,
    turns: list[list[int]],
    count: Callable[[list], int],
    budget: int,
) -> int | None:
    """Fewest oldest turns to drop so count(items) <= budget, or None when even
    keeping only the latest turn does not fit.  count() is monotonic in the
    number of dropped turns, so bisect on exact counts."""
    max_drop = len(turns) - 1
    if max_drop < 1 or count(_drop_turns(items, turns, max_drop)) > budget:
        return None
    lo, hi = 1, max_drop
    while lo < hi:
        mid = (lo + hi) // 2
        if count(_drop_turns(items, turns, mid)) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ── input-shrinking steps ───────────────────────────────────────────────────
# Items are chat messages ("chat") or Responses input items ("responses").  Both
# are plain dicts; steps return a new list, or None when nothing changed.

def _get(item: Any, path: tuple) -> Any:
    for key in path:
        item = item[key]
    return item


def _set(item: Any, path: tuple, value: Any) -> Any:
    """Copy of `item` with the value at `path` replaced (copies only the spine)."""
    if not path:
        return value
    child = _set(item[path[0]], path[1:], value)
    if isinstance(item, list):
        out = list(item)
    else:
        out = dict(item)
    out[path[0]] = child
    return out


def _content_paths(item: dict, key: str, kind: str):
    content = item.get(key)
    if isinstance(content, str):
        yield kind, (key,)
    elif isinstance(content, list):
        for i, part in enumerate(content):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield kind, (key, i, "text")


def _text_paths(item: Any, shape: str):
    """(kind, path) of every text field in an item: "message" (user/assistant
    text), "result" (tool output) or "args" (tool-call arguments JSON)."""
    if not isinstance(item, dict) or item.get("role") in _FIXED_ROLES:
        return
    if shape == "chat":
        if item.get("role") == "tool":
            yield from _content_paths(item, "content", "result")
            return
        yield from _content_paths(item, "content", "message")
        for i, call in enumerate(item.get("tool_calls") or []):
            if isinstance((call.get("function") or {}).get("arguments"), str):
                yield "args", ("tool_calls", i, "function", "arguments")
        return
    kind = item.get("type")
    if kind == "function_call_output":
        yield from _content_paths(item, "output", "result")
    elif kind == "function_call":
        if isinstance(item.get("arguments"), str):
            yield "args", ("arguments",)
    elif kind != "reasoning":
        yield from _content_paths(item, "content", "message")


def _history_end(items: list, role_of: Callable[[Any], str | None]) -> int:
    """Index where the latest turn starts; everything before it is history."""
    turns = _split_turns(items, role_of)
    return turns[-1][0] if turns else 0


def _without_thinking(text: str) -> str:
    """Drop a leading chain of thought.  Qwen's generation prompt already opens
    <think>, so a stored answer carries only the closing tag."""
    return text.rsplit(_THINK_END, 1)[1].lstrip() if _THINK_END in text else text


def _strip_thinking(items: list, shape: str, role_of, limits: ModelLimits) -> list | None:
    end = _history_end(items, role_of)
    out: list = []
    changed = False
    for i, item in enumerate(items):
        if i >= end or not isinstance(item, dict):
            out.append(item)
            continue
        if shape == "responses" and item.get("type") == "reasoning":
            changed = True
            continue
        if shape == "chat" and item.get("role") == "assistant":
            stripped = {k: v for k, v in item.items() if k not in ("reasoning_content", "thinking_blocks", "reasoning")}
            changed |= len(stripped) != len(item)
            item = stripped
        if item.get("role") == "assistant":
            for kind, path in _text_paths(item, shape):
                if kind == "message":
                    text = _get(item, path)
                    plain = _without_thinking(text)
                    if plain != text:
                        item, changed = _set(item, path, plain), True
        out.append(item)
    return out if changed else None


def _shingles(text: str) -> set[int]:
    words = text.split()
    if len(words) < _SHINGLE_WORDS:
        return {hash(tuple(words))}
    return {hash(tuple(words[i:i + _SHINGLE_WORDS])) for i in range(len(words) - _SHINGLE_WORDS + 1)}


def _dedup_similar(items: list, shape: str, role_of, limits: ModelLimits) -> list | None:
    """Stub history messages that are >= _DEDUP_SIMILARITY similar to a later one.
    Walks from the end so the latest copy is the one kept."""
    end = _history_end(items, role_of)
    out = list(items)
    kept: list[tuple[int, set[int]]] = []
    changed = False
    for idx in range(len(items) - 1, -1, -1):
        for kind, path in _text_paths(items[idx], shape):
            if kind == "args":
                continue
            text = _get(items[idx], path)
            if len(text) < _DEDUP_MIN_CHARS:
                continue
            grams = _shingles(text)
            # Jaccard >= s needs the sizes to be within a factor s of each other.
            duplicate = any(
                min(len(grams), len(other)) >= _DEDUP_SIMILARITY * max(len(grams), len(other))
                and len(grams & other) / len(grams | other) >= _DEDUP_SIMILARITY
                for _, other in kept
            )
            if duplicate and idx < end:
                out[idx] = _set(out[idx], path, _DUPLICATE_STUB)
                changed = True
            else:
                kept.append((len(text), grams))
    return out if changed else None


def _cap_text(text: str, cap: int, tokenizer: _Counter) -> str:
    """Keep the first and last cap/2 tokens of `text`."""
    if len(text) <= cap:           # a token is at least one character
        return text
    ids = tokenizer.encode(text)
    if len(ids) <= cap:
        return text
    head = cap // 2
    return (
        tokenizer.decode(ids[:head])
        + f"\n[... {len(ids) - cap} tokens omitted ...]\n"
        + tokenizer.decode(ids[len(ids) - (cap - head):])
    )


def _cap_json_strings(value: Any, cap: int, tokenizer: _Counter) -> Any:
    if isinstance(value, str):
        return _cap_text(value, cap, tokenizer)
    if isinstance(value, list):
        return [_cap_json_strings(v, cap, tokenizer) for v in value]
    if isinstance(value, dict):
        return {k: _cap_json_strings(v, cap, tokenizer) for k, v in value.items()}
    return value


def _cap_tool_io(items: list, shape: str, role_of, limits: ModelLimits) -> list | None:
    """Cap history tool results, and the string values inside tool-call
    arguments (which must stay valid JSON, so the raw string is never cut)."""
    end = _history_end(items, role_of)
    cap, tokenizer = limits.tool_cap_tokens, limits.tokenizer
    out = list(items)
    changed = False
    for idx in range(end):
        for kind, path in _text_paths(items[idx], shape):
            if kind == "message":
                continue
            text = _get(items[idx], path)
            if kind == "args":
                try:
                    capped = json.dumps(_cap_json_strings(json.loads(text), cap, tokenizer), ensure_ascii=False)
                except json.JSONDecodeError:
                    continue
                if capped == json.dumps(json.loads(text), ensure_ascii=False):
                    continue
            else:
                capped = _cap_text(text, cap, tokenizer)
                if capped == text:
                    continue
            out[idx] = _set(out[idx], path, capped)
            changed = True
    return out if changed else None


class ContextTrimHook(CustomLogger):
    """Fit input + output into the context window: shrink the output cap, and
    drop the oldest turns when shrinking alone would cut the answer too much."""

    def __init__(
        self,
        config_path: str = _CONFIG_PATH,
        tokenizer_loader: Callable[[str, str | None], Any] = _load_tokenizer,
    ) -> None:
        super().__init__()
        self._limits = _load_model_limits(config_path, tokenizer_loader)
        logger.info("trim_hook: limits for %s", {k: v.context_window for k, v in self._limits.items()})

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        if call_type == "aresponses":
            return await self._responses(data)
        if call_type in ("completion", "acompletion"):
            return await self._chat(data)
        return None

    # ── chat/completions ────────────────────────────────────────────────────

    async def _chat(self, data: dict) -> dict | None:
        messages: list[dict] | None = data.get("messages")
        if not messages:
            return None

        # Same Qwen "No user query found in messages" guard as the Responses path.
        injected = False
        if not _has_user_role(messages):
            messages = list(messages) + [{"role": "user", "content": _FILLER_USER}]
            data["messages"] = messages
            injected = True
            logger.warning(
                "trim_hook: chat request for %s had no user turn (%d msgs) — "
                "appended filler user turn to satisfy the chat template",
                data.get("model"), len(messages) - 1,
            )

        limits = self._limits.get((data.get("model") or "").rsplit("/", 1)[-1])
        if limits is None:
            return data if injected else None

        tools = data.get("tools") or None
        changed = await asyncio.to_thread(
            self._fit, data, limits, messages, "messages", "chat",
            _CHAT_OUTPUT_KEYS, lambda m: m.get("role"),
            lambda items: limits.tokenizer.count(items, tools),
        )
        return data if (changed or injected) else None

    # ── responses ───────────────────────────────────────────────────────────

    async def _responses(self, data: dict) -> dict | None:
        injected = self._ensure_responses_user_turn(data)

        limits = self._limits.get((data.get("model") or "").rsplit("/", 1)[-1])
        if limits is None:
            return data if injected else None

        # Imported here: litellm.responses pulls in the whole proxy stack.
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig as Bridge,
        )

        # LiteLLM prepends the stored conversation for previous_response_id
        # downstream of this hook.  It cannot be trimmed here, but it occupies
        # context, so it has to be part of the count.
        history: list = []
        if data.get("previous_response_id"):
            from previous_response_hook import load_session_messages

            history = await load_session_messages(data["previous_response_id"])

        tools, _ = Bridge.transform_responses_api_tools_to_chat_completion_tools(data.get("tools") or [])
        instructions = {"instructions": data.get("instructions")}

        def count(items: list | str) -> int:
            messages = Bridge.transform_responses_api_input_to_messages(items, instructions)
            return limits.tokenizer.count(list(history) + list(messages), tools)

        changed = await asyncio.to_thread(
            self._fit, data, limits, data.get("input"), "input", "responses",
            _RESPONSES_OUTPUT_KEYS, lambda item: item.get("role") if isinstance(item, dict) else None,
            count,
        )
        return data if (changed or injected) else None

    # ── shared policy ───────────────────────────────────────────────────────

    def _fit(
        self,
        data: dict,
        limits: ModelLimits,
        items: Any,
        items_key: str,
        shape: str,
        output_keys: tuple[str, ...],
        role_of: Callable[[Any], str | None],
        count: Callable[[Any], int],
    ) -> bool:
        """Apply the module-docstring policy to `data` in place; True if changed."""
        model = data.get("model")
        requested = next((data[k] for k in output_keys if data.get(k)), None) or limits.default_output
        floor = math.ceil(requested / _OUTPUT_TOLERANCE)
        ctx = limits.context_window

        input_tokens = count(items)
        room = ctx - input_tokens - _SAFETY_MARGIN
        changed = False

        if room < floor and isinstance(items, list):
            for name, step in (
                ("thinking", _strip_thinking),
                ("near-duplicates", _dedup_similar),
                ("tool arguments/results", _cap_tool_io),
            ):
                shrunk = step(items, shape, role_of, limits)
                if shrunk is None:
                    continue
                new_tokens = count(shrunk)
                logger.info(
                    "trim_hook: shrank %s for %s (input tokens %d → %d, ctx=%d, requested output=%d)",
                    name, model, input_tokens, new_tokens, ctx, requested,
                )
                items, input_tokens, room, changed = shrunk, new_tokens, ctx - new_tokens - _SAFETY_MARGIN, True
                data[items_key] = items
                if room >= floor:
                    break

        if room < floor and isinstance(items, list):
            turns = _split_turns(items, role_of)
            for target in (floor, min(_MIN_OUTPUT, floor)):
                if room >= target:
                    break
                n_drop = _fit_input(items, turns, count, ctx - _SAFETY_MARGIN - target)
                if n_drop is None:
                    continue
                trimmed = _drop_turns(items, turns, n_drop)
                new_tokens = count(trimmed)
                logger.warning(
                    "trim_hook: dropped %d oldest turn(s) for %s "
                    "(input tokens %d → %d, items %d → %d, ctx=%d, requested output=%d)",
                    n_drop, model, input_tokens, new_tokens, len(items), len(trimmed), ctx, requested,
                )
                data[items_key] = trimmed
                input_tokens, room, changed = new_tokens, ctx - new_tokens - _SAFETY_MARGIN, True
                break

        if room < 1:
            logger.warning(
                "trim_hook: %s input is %d tokens, no room left in ctx=%d — "
                "leaving the request for vLLM to reject",
                model, input_tokens, ctx,
            )
            return changed

        if requested > room:
            logger.info(
                "trim_hook: capped output %d → %d for %s (input tokens=%d, ctx=%d)",
                requested, room, model, input_tokens, ctx,
            )
            present = [k for k in output_keys if data.get(k)]
            for key in present or output_keys[:1]:
                data[key] = room
            changed = True
        return changed

    def _ensure_responses_user_turn(self, data: dict) -> bool:
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
                return False           # non-empty string is the user turn
            items: list[dict] = []     # empty string → treat as no user turn
        elif isinstance(input_val, list):
            if _has_user_role(input_val):
                return False
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
        return True


# Module-level instance — config.yaml references this via "trim_hook.context_trim_hook"
context_trim_hook = ContextTrimHook()
