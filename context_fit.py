"""Fit a /v1/chat/completions request into the model's context window.

Runs in model_manager, the last hop before vLLM.  vLLM rejects a request whose
prompt + max output exceeds --max-model-len with a 400, so the gateway has to
make the request fit.  Doing it here (not in a LiteLLM hook) means it sees the
final prompt: LiteLLM's Responses bridge appends the stored previous_response_id
history after its pre-call hooks, so a hook can count that history but never trim
it, whereas the chat/completions body that arrives here already contains it.

Policy, with R = context_window - input_tokens - SAFETY_MARGIN (room left for
the answer) and F = requested_output / OUTPUT_TOLERANCE (the least room we
accept without touching the input):

  * R >= requested_output            nothing to do.
  * F <= R < requested_output        shrink the output cap to R.  The model
                                     stops at the cap with finish_reason
                                     "length", which is a normal response, not
                                     an error.
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
                                     user message; system/developer messages and
                                     the latest turn are never modified or
                                     dropped.  Then the output cap shrinks to what
                                     is left.
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
still undercount by up to SAFETY_MARGIN-scale amounts per image.

Only bodies with a `messages` list are touched; models without
model_info.max_input_tokens (TTS, ASR) have no limits and pass through.
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

logger = logging.getLogger("mgr.context_fit")

# Single source of truth is config.yaml, the same file LiteLLM loads.  Per model:
#   model_info.max_input_tokens   context window (== vLLM --max-model-len); LiteLLM
#                                 also registers it into litellm.model_cost
#   litellm_params.max_tokens     output cap used when the request carries none
#   model_info.tokenizer(+_revision)  HF repo of the tokenizer vLLM serves with
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
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
_OUTPUT_KEYS = ("max_tokens", "max_completion_tokens")


class ModelLimits(NamedTuple):
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


def _template_messages(messages: list[dict]) -> list[dict]:
    """Messages as vLLM hands them to the chat template: tool-call arguments
    arrive as JSON strings on the wire but the template iterates them as dicts."""
    out = []
    for msg in messages:
        if msg.get("tool_calls"):
            calls = []
            for call in msg["tool_calls"]:
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


def load_model_limits(
    path: str, tokenizer_loader: Callable[[str, str | None], Any] = _load_tokenizer
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
            raise ValueError(f"context_fit: model {name!r} declares max_input_tokens but no model_info.tokenizer")
        max_out = (entry.get("litellm_params") or {}).get("max_tokens")
        limits[name] = ModelLimits(
            context_window=int(ctx),
            default_output=int(max_out) if max_out else _DEFAULT_OUT,
            tokenizer=_Counter(tokenizer_loader(repo, info.get("tokenizer_revision"))),
            tool_cap_tokens=int(info.get("tool_cap_tokens") or _DEFAULT_TOOL_CAP_TOKENS),
        )
    return limits


def _role(msg: Any) -> str | None:
    return msg.get("role") if isinstance(msg, dict) else None


def _has_user_role(messages: list) -> bool:
    return any(_role(m) == "user" for m in messages)


def _split_turns(messages: list) -> list[list[int]]:
    """Indices of `messages` grouped into droppable turns.  A turn opens at each
    user message; everything up to the next one (assistant replies, tool calls
    and their results) belongs to it.  System/developer messages are never in a
    turn.  Messages before the first user message form their own leading turn."""
    turns: list[list[int]] = []
    for i, msg in enumerate(messages):
        role = _role(msg)
        if role in _FIXED_ROLES:
            continue
        if role == "user" or not turns:
            turns.append([i])
        else:
            turns[-1].append(i)
    return turns


def _drop_turns(messages: list, turns: list[list[int]], n: int) -> list:
    dropped = {i for turn in turns[:n] for i in turn}
    return [msg for i, msg in enumerate(messages) if i not in dropped]


def _fit_input(
    messages: list,
    turns: list[list[int]],
    count: Callable[[list], int],
    budget: int,
) -> int | None:
    """Fewest oldest turns to drop so count(messages) <= budget, or None when even
    keeping only the latest turn does not fit.  count() is monotonic in the
    number of dropped turns, so bisect on exact counts."""
    max_drop = len(turns) - 1
    if max_drop < 1 or count(_drop_turns(messages, turns, max_drop)) > budget:
        return None
    lo, hi = 1, max_drop
    while lo < hi:
        mid = (lo + hi) // 2
        if count(_drop_turns(messages, turns, mid)) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ── input-shrinking steps ───────────────────────────────────────────────────
# Each step returns a new message list, or None when nothing changed.

def _get(item: Any, path: tuple) -> Any:
    for key in path:
        item = item[key]
    return item


def _set(item: Any, path: tuple, value: Any) -> Any:
    """Copy of `item` with the value at `path` replaced (copies only the spine)."""
    if not path:
        return value
    child = _set(item[path[0]], path[1:], value)
    out = list(item) if isinstance(item, list) else dict(item)
    out[path[0]] = child
    return out


def _content_paths(msg: dict, kind: str):
    content = msg.get("content")
    if isinstance(content, str):
        yield kind, ("content",)
    elif isinstance(content, list):
        for i, part in enumerate(content):
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                yield kind, ("content", i, "text")


def _text_paths(msg: Any):
    """(kind, path) of every text field in a message: "message" (user/assistant
    text), "result" (tool output) or "args" (tool-call arguments JSON)."""
    if not isinstance(msg, dict) or msg.get("role") in _FIXED_ROLES:
        return
    if msg.get("role") == "tool":
        yield from _content_paths(msg, "result")
        return
    yield from _content_paths(msg, "message")
    for i, call in enumerate(msg.get("tool_calls") or []):
        if isinstance((call.get("function") or {}).get("arguments"), str):
            yield "args", ("tool_calls", i, "function", "arguments")


def _history_end(messages: list) -> int:
    """Index where the latest turn starts; everything before it is history."""
    turns = _split_turns(messages)
    return turns[-1][0] if turns else 0


def _without_thinking(text: str) -> str:
    """Drop a leading chain of thought.  Qwen's generation prompt already opens
    <think>, so a stored answer carries only the closing tag."""
    return text.rsplit(_THINK_END, 1)[1].lstrip() if _THINK_END in text else text


def _strip_thinking(messages: list, limits: ModelLimits) -> list | None:
    end = _history_end(messages)
    out: list = []
    changed = False
    for i, msg in enumerate(messages):
        if i >= end or _role(msg) != "assistant":
            out.append(msg)
            continue
        stripped = {k: v for k, v in msg.items() if k not in ("reasoning_content", "thinking_blocks", "reasoning")}
        changed |= len(stripped) != len(msg)
        msg = stripped
        for kind, path in _text_paths(msg):
            if kind == "message":
                text = _get(msg, path)
                plain = _without_thinking(text)
                if plain != text:
                    msg, changed = _set(msg, path, plain), True
        out.append(msg)
    return out if changed else None


def _shingles(text: str) -> set[int]:
    words = text.split()
    if len(words) < _SHINGLE_WORDS:
        return {hash(tuple(words))}
    return {hash(tuple(words[i:i + _SHINGLE_WORDS])) for i in range(len(words) - _SHINGLE_WORDS + 1)}


def _dedup_similar(messages: list, limits: ModelLimits) -> list | None:
    """Stub history messages that are >= _DEDUP_SIMILARITY similar to a later one.
    Walks from the end so the latest copy is the one kept."""
    end = _history_end(messages)
    out = list(messages)
    kept: list[tuple[int, set[int]]] = []
    changed = False
    for idx in range(len(messages) - 1, -1, -1):
        for kind, path in _text_paths(messages[idx]):
            if kind == "args":
                continue
            text = _get(messages[idx], path)
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


def _cap_tool_io(messages: list, limits: ModelLimits) -> list | None:
    """Cap history tool results, and the string values inside tool-call
    arguments (which must stay valid JSON, so the raw string is never cut)."""
    end = _history_end(messages)
    cap, tokenizer = limits.tool_cap_tokens, limits.tokenizer
    out = list(messages)
    changed = False
    for idx in range(end):
        for kind, path in _text_paths(messages[idx]):
            if kind == "message":
                continue
            text = _get(messages[idx], path)
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


class ContextFitter:
    """Fit input + output into the context window: shrink the output cap, and
    shrink or drop history when shrinking the cap alone would cut the answer
    too much."""

    def __init__(self, limits: dict[str, ModelLimits]) -> None:
        self._limits = limits

    @classmethod
    def from_config(
        cls,
        path: str = CONFIG_PATH,
        tokenizer_loader: Callable[[str, str | None], Any] = _load_tokenizer,
    ) -> "ContextFitter":
        limits = load_model_limits(path, tokenizer_loader)
        logger.info("context_fit: limits for %s", {k: v.context_window for k, v in limits.items()})
        return cls(limits)

    async def fit(self, body: dict, model_name: str) -> bool:
        """Apply the module-docstring policy to a parsed chat/completions body in
        place; True if it changed.  Tokenizing and rendering the chat template is
        CPU work, so it runs in a thread."""
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            return False

        injected = False
        if not _has_user_role(messages):
            body["messages"] = messages = messages + [{"role": "user", "content": _FILLER_USER}]
            injected = True
            logger.warning(
                "context_fit: chat request for %s had no user turn (%d msgs) — "
                "appended filler user turn to satisfy the chat template",
                model_name, len(messages) - 1,
            )

        limits = self._limits.get(model_name)
        if limits is None:
            return injected
        changed = await asyncio.to_thread(self._fit, body, model_name, limits)
        return changed or injected

    def _fit(self, data: dict, model: str, limits: ModelLimits) -> bool:
        tools = data.get("tools") or None
        messages = data["messages"]

        def count(items: list) -> int:
            return limits.tokenizer.count(items, tools)

        requested = next((data[k] for k in _OUTPUT_KEYS if data.get(k)), None) or limits.default_output
        floor = math.ceil(requested / _OUTPUT_TOLERANCE)
        ctx = limits.context_window

        input_tokens = count(messages)
        room = ctx - input_tokens - _SAFETY_MARGIN
        changed = False

        if room < floor:
            for name, step in (
                ("thinking", _strip_thinking),
                ("near-duplicates", _dedup_similar),
                ("tool arguments/results", _cap_tool_io),
            ):
                shrunk = step(messages, limits)
                if shrunk is None:
                    continue
                new_tokens = count(shrunk)
                logger.info(
                    "context_fit: shrank %s for %s (input tokens %d → %d, ctx=%d, requested output=%d)",
                    name, model, input_tokens, new_tokens, ctx, requested,
                )
                messages, input_tokens, room, changed = shrunk, new_tokens, ctx - new_tokens - _SAFETY_MARGIN, True
                data["messages"] = messages
                if room >= floor:
                    break

        if room < floor:
            turns = _split_turns(messages)
            for target in (floor, min(_MIN_OUTPUT, floor)):
                if room >= target:
                    break
                n_drop = _fit_input(messages, turns, count, ctx - _SAFETY_MARGIN - target)
                if n_drop is None:
                    continue
                trimmed = _drop_turns(messages, turns, n_drop)
                new_tokens = count(trimmed)
                logger.warning(
                    "context_fit: dropped %d oldest turn(s) for %s "
                    "(input tokens %d → %d, messages %d → %d, ctx=%d, requested output=%d)",
                    n_drop, model, input_tokens, new_tokens, len(messages), len(trimmed), ctx, requested,
                )
                data["messages"] = trimmed
                input_tokens, room, changed = new_tokens, ctx - new_tokens - _SAFETY_MARGIN, True
                break

        if room < 1:
            logger.warning(
                "context_fit: %s input is %d tokens, no room left in ctx=%d — "
                "leaving the request for vLLM to reject",
                model, input_tokens, ctx,
            )
            return changed

        if requested > room:
            logger.info(
                "context_fit: capped output %d → %d for %s (input tokens=%d, ctx=%d)",
                requested, room, model, input_tokens, ctx,
            )
            present = [k for k in _OUTPUT_KEYS if data.get(k)]
            for key in present or _OUTPUT_KEYS[:1]:
                data[key] = room
            changed = True
        return changed
