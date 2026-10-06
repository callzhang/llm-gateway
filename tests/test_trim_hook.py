"""trim_hook: output-cap shrink + oldest-turn drop for chat and Responses.

Unit tests use a stub tokenizer whose token count is exact and easy to reason
about (1 token per content character + 4 per message + tool-definition length).
The integration tests at the bottom run the real Qwen tokenizer and the real
config.yaml, including the request shape that failed in production
(98,305 prompt tokens + 32,768 requested output > 131,072).
"""
import asyncio
import json
import math
import os
from unittest.mock import AsyncMock, patch

import pytest
import yaml

import trim_hook
from trim_hook import ContextTrimHook

CTX = 10_000
MARGIN = trim_hook._SAFETY_MARGIN
CONFIG = {
    "model_list": [
        {
            "model_name": "chat-model",
            "litellm_params": {"model": "custom_openai/chat-model", "max_tokens": 3000},
            "model_info": {"max_input_tokens": CTX, "tokenizer": "stub/tok", "tool_cap_tokens": 500},
        },
        {"model_name": "tts-model", "litellm_params": {"model": "custom_openai/tts"}},
    ]
}


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text)

    def decode(self, ids):
        return "".join(ids)

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=True, return_dict=True):
        n = 0
        for m in messages:
            content = m.get("content") or ""
            if isinstance(content, list):
                content = "".join(p.get("text", "") for p in content)
            n += len(content) + 4 + len(m.get("reasoning_content") or "")
            n += len(json.dumps(m.get("tool_calls") or ""))
        if tools:
            n += len(json.dumps(tools))
        return {"input_ids": [0] * n}


@pytest.fixture
def hook(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG))
    return ContextTrimHook(config_path=str(path), tokenizer_loader=lambda repo, rev: StubTokenizer())


def run(hook, data, call_type="acompletion"):
    return asyncio.run(hook.async_pre_call_hook(None, None, data, call_type))


def tokens(messages):
    return StubTokenizer().apply_chat_template(messages)["input_ids"].__len__()


def turn(i, size):
    return [{"role": "user", "content": f"u{i}".ljust(size, "x")}, {"role": "assistant", "content": f"a{i}".ljust(size, "y")}]


# ── output cap ──────────────────────────────────────────────────────────────

def test_request_that_fits_is_untouched(hook):
    data = {"model": "chat-model", "messages": [{"role": "user", "content": "x" * 100}], "max_tokens": 500}
    assert run(hook, data) is None
    assert data["max_tokens"] == 500


def test_overflow_within_tolerance_shrinks_output_without_trimming(hook):
    history = turn(0, 500)
    last = {"role": "user", "content": "z" * 5500}
    messages = history + [last]
    data = {"model": "chat-model", "messages": list(messages), "max_tokens": 3000}
    room = CTX - tokens(messages) - MARGIN
    assert math.ceil(3000 / 1.1) <= room < 3000

    assert run(hook, data) is data
    assert data["max_tokens"] == room
    assert data["messages"] == messages


def test_default_output_cap_applies_when_request_has_none(hook):
    data = {"model": "chat-model", "messages": [{"role": "user", "content": "z" * 7000}]}
    room = CTX - tokens(data["messages"]) - MARGIN
    assert room < 3000

    run(hook, data)
    assert data["max_tokens"] == room


def test_max_completion_tokens_is_the_field_that_gets_capped(hook):
    data = {"model": "chat-model", "messages": [{"role": "user", "content": "z" * 7000}], "max_completion_tokens": 3000}
    run(hook, data)
    assert data["max_completion_tokens"] == CTX - tokens(data["messages"]) - MARGIN
    assert "max_tokens" not in data


def test_tool_definitions_count_toward_the_prompt(hook):
    tools = [{"type": "function", "function": {"name": "f", "description": "d" * 6500}}]
    messages = [{"role": "user", "content": "hi"}]
    data = {"model": "chat-model", "messages": messages, "tools": tools, "max_tokens": 3000}

    run(hook, data)
    assert data["max_tokens"] == CTX - tokens(messages) - len(json.dumps(tools)) - MARGIN


# ── trimming ────────────────────────────────────────────────────────────────

def test_large_overflow_drops_oldest_turns_and_keeps_system_and_last_turn(hook):
    system = {"role": "system", "content": "sys"}
    history = [m for i in range(6) for m in turn(i, 1000)]
    last = {"role": "user", "content": "last question"}
    data = {"model": "chat-model", "messages": [system] + history + [last], "max_tokens": 3000}
    floor = math.ceil(3000 / 1.1)

    assert run(hook, data) is data
    kept = data["messages"]
    assert kept[0] == system and kept[-1] == last
    assert kept[1]["content"].startswith("u") and kept[1]["role"] == "user"   # whole turns only
    assert len(kept) < len(history) + 2
    assert CTX - tokens(kept) - MARGIN >= floor
    # fewest turns dropped: one more turn back would no longer fit the floor
    older = [system] + history[len(history) - (len(kept) - 2) - 2:] + [last]
    assert CTX - tokens(older) - MARGIN < floor
    assert data["max_tokens"] == 3000


def test_tool_call_and_its_result_are_dropped_together(hook):
    call = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    result = {"role": "tool", "tool_call_id": "c1", "content": "r" * 3000}
    messages = [
        {"role": "user", "content": "first"}, call, result, {"role": "assistant", "content": "done"},
        {"role": "user", "content": "q" * 8000},
    ]
    data = {"model": "chat-model", "messages": messages, "max_tokens": 3000}

    run(hook, data)
    assert [m["role"] for m in data["messages"]] == ["user"]


def test_history_is_kept_when_dropping_all_of_it_still_cannot_reach_the_floor(hook):
    history = turn(0, 100)
    last = {"role": "user", "content": "z" * 8000}
    data = {"model": "chat-model", "messages": history + [last], "max_tokens": 3000}
    room = CTX - tokens(history + [last]) - MARGIN
    assert 0 < room < math.ceil(3000 / 1.1)
    assert CTX - tokens([last]) - MARGIN < math.ceil(3000 / 1.1)

    run(hook, data)
    assert data["messages"] == history + [last]
    assert data["max_tokens"] == room


def test_history_is_dropped_only_as_far_as_the_minimum_output_when_the_floor_is_unreachable(hook):
    # Full input overflows, and even the latest turn alone leaves less than the
    # 2728-token floor, but more than the 1024 minimum: drop just enough to be servable.
    history = [m for i in range(3) for m in turn(i, 1000)]
    last = {"role": "user", "content": "z" * 7000}
    data = {"model": "chat-model", "messages": history + [last], "max_tokens": 3000}

    run(hook, data)
    kept = data["messages"]
    assert kept[-1] == last
    assert CTX - tokens(kept) - MARGIN >= trim_hook._MIN_OUTPUT
    assert data["max_tokens"] == CTX - tokens(kept) - MARGIN



# ── shrinking steps before dropping turns ───────────────────────────────────
# Budget with the stub tokenizer: input must end up <= CTX - MARGIN - 2728 = 6760.

THINK = "pondering " * 300          # 3000 chars of chain of thought


def calm_last(size=2000):
    return {"role": "user", "content": "final question".ljust(size, "?")}


def test_thinking_before_the_latest_user_message_is_removed_first(hook):
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": THINK},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": THINK + "</think>\n\na2"},
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3", "thinking_blocks": [{"thinking": THINK}]},
    ]
    last = calm_last(2000)
    data = {"model": "chat-model", "messages": history + [last], "max_tokens": 3000}
    assert tokens(history + [last]) > 6760

    assert run(hook, data) is data
    kept = data["messages"]
    assert [m["content"] for m in kept[:6]] == ["q1", "a1", "q2", "a2", "q3", "a3"]
    assert not any(k in m for m in kept for k in ("reasoning_content", "thinking_blocks"))
    assert len(kept) == len(history) + 1          # nothing dropped
    assert kept[-1] == last


def test_thinking_in_the_current_turn_is_kept(hook):
    last = {"role": "user", "content": "go".ljust(9000, ".")}
    reply = {"role": "assistant", "content": "", "reasoning_content": THINK,
             "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    data = {"model": "chat-model", "messages": [last, reply, {"role": "tool", "tool_call_id": "c", "content": "ok"}], "max_tokens": 3000}
    run(hook, data)
    assert data["messages"][1]["reasoning_content"] == THINK


def test_responses_reasoning_items_and_inline_thinking_are_removed(hook):
    items = [
        {"role": "user", "content": "q1"},
        {"type": "reasoning", "id": "r1", "summary": [{"type": "summary_text", "text": THINK}]},
        {"role": "assistant", "content": [{"type": "output_text", "text": THINK + "</think>\n\nanswer"}]},
        {"role": "user", "content": "q2".ljust(7000, "?")},
    ]
    data = {"model": "chat-model", "input": items, "max_output_tokens": 3000}
    run(hook, data, "aresponses")
    assert [i.get("type") for i in data["input"]] == [None, None, None]
    assert data["input"][1]["content"][0]["text"] == "answer"


def test_later_steps_are_skipped_once_the_request_fits(hook):
    dup = "same file contents " * 40
    history = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1", "reasoning_content": THINK},
        {"role": "user", "content": dup},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": dup},
        {"role": "assistant", "content": "ok again"},
    ]
    last = calm_last(2300)
    data = {"model": "chat-model", "messages": history + [last], "max_tokens": 3000}
    assert tokens(history + [last]) - len(THINK) <= 6760 < tokens(history + [last])

    run(hook, data)
    assert [m["content"] for m in data["messages"] if m["role"] == "user"][1:3] == [dup, dup]   # dedup not run


def test_near_duplicate_history_is_stubbed_and_the_latest_copy_kept(hook):
    page = " ".join(f"word{i}" for i in range(600))                 # ~4.1k chars
    nearly = page.replace("word300", "WORD300")                      # 1 word changed: 591/601 = 98.3% of 5-grams shared
    different = " ".join(f"other{i}" for i in range(60))             # ~0.4k chars, unrelated
    history = [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": page},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": different},
        {"role": "user", "content": "q3"}, {"role": "assistant", "content": nearly},
    ]
    last = {"role": "user", "content": "go".ljust(1500, "?")}
    data = {"model": "chat-model", "messages": history + [last], "max_tokens": 3000}
    assert tokens(history + [last]) > 6760

    run(hook, data)
    got = [m["content"] for m in data["messages"]]
    assert got[1] == trim_hook._DUPLICATE_STUB        # older copy replaced
    assert got[3] == different                        # dissimilar kept
    assert got[5] == nearly                           # latest copy kept
    assert [m["role"] for m in data["messages"]] == [m["role"] for m in history + [last]]


def test_a_copy_in_the_latest_turn_is_kept_and_the_history_copy_is_stubbed(hook):
    page = " ".join(f"word{i}" for i in range(600))
    messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": page}, {"role": "user", "content": page}]
    data = {"model": "chat-model", "messages": list(messages), "max_tokens": 3000}
    assert tokens(messages) > 6760

    run(hook, data)
    assert data["messages"][1]["content"] == trim_hook._DUPLICATE_STUB
    assert data["messages"][2] == messages[2]


def test_a_message_that_is_less_similar_than_the_threshold_is_kept(hook):
    page = " ".join(f"word{i}" for i in range(600))
    edited = " ".join(w if i % 20 else "changed" for i, w in enumerate(page.split()))   # every 20th word differs
    messages = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": page},
                {"role": "user", "content": "q2"}, {"role": "assistant", "content": edited},
                {"role": "user", "content": "go".ljust(500, "?")}]
    data = {"model": "chat-model", "messages": list(messages), "max_tokens": 3000}
    run(hook, data)
    assert trim_hook._DUPLICATE_STUB not in [m["content"] for m in data["messages"]]


def test_short_messages_are_never_deduplicated(hook):
    history = [m for i in range(10) for m in ({"role": "user", "content": "same"}, {"role": "assistant", "content": "same"})]
    messages = history + [{"role": "user", "content": "z" * 3000}]
    data = {"model": "chat-model", "messages": messages, "max_tokens": 3000}
    run(hook, data)
    assert trim_hook._DUPLICATE_STUB not in [m["content"] for m in data["messages"]]


def test_history_tool_results_and_arguments_are_capped_but_the_current_ones_are_not(hook):
    call = lambda cid, args: {"role": "assistant", "content": "", "tool_calls": [
        {"id": cid, "type": "function", "function": {"name": "f", "arguments": args}}]}
    args_old = json.dumps({"path": "a.txt", "body": "B" * 3000, "n": 7})
    args_new = json.dumps({"body": "N" * 1500})
    messages = [
        {"role": "user", "content": "first"}, call("c1", args_old), {"role": "tool", "tool_call_id": "c1", "content": "R" * 3000},
        {"role": "user", "content": "second".ljust(300, "?")}, call("c2", args_new),
        {"role": "tool", "tool_call_id": "c2", "content": "S" * 1500},
    ]
    data = {"model": "chat-model", "messages": messages, "max_tokens": 3000}
    assert tokens(messages) > 6760

    run(hook, data)
    kept = data["messages"]
    assert len(kept) == len(messages)
    assert kept[2]["content"].startswith("R" * 250) and kept[2]["content"].endswith("R" * 250)
    assert "tokens omitted" in kept[2]["content"] and len(kept[2]["content"]) < 700
    old_args = json.loads(kept[1]["tool_calls"][0]["function"]["arguments"])   # still valid JSON
    assert old_args["path"] == "a.txt" and old_args["n"] == 7
    assert "tokens omitted" in old_args["body"] and len(old_args["body"]) < 700
    assert kept[4] == messages[4] and kept[5] == messages[5]                  # current turn untouched


def test_responses_function_call_items_are_capped_in_history(hook):
    items = [
        {"role": "user", "content": "first"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": json.dumps({"body": "B" * 3000})},
        {"type": "function_call_output", "call_id": "c1", "output": "R" * 3000},
        {"role": "user", "content": "second".ljust(5000, "?")},
    ]
    data = {"model": "chat-model", "input": items, "max_output_tokens": 3000}
    run(hook, data, "aresponses")
    kept = data["input"]
    assert len(kept) == len(items)
    assert "tokens omitted" in kept[2]["output"]
    assert "tokens omitted" in json.loads(kept[1]["arguments"])["body"]


def test_tool_arguments_that_are_not_json_are_left_alone(hook):
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {"name": "f", "arguments": "not json " * 500}}]},
        {"role": "user", "content": "q".ljust(8000, "?")},
    ]
    data = {"model": "chat-model", "messages": msgs, "max_tokens": 3000}
    run(hook, data)
    kept_args = [m for m in data["messages"] if m.get("tool_calls")]
    assert not kept_args or kept_args[0]["tool_calls"][0]["function"]["arguments"] == "not json " * 500


def test_turns_are_dropped_only_after_every_shrinking_step_was_not_enough(hook):
    history = [m for i in range(5) for m in turn(i, 1500)]       # plain text, nothing to strip/dedup/cap
    data = {"model": "chat-model", "messages": history + [calm_last(100)], "max_tokens": 3000}
    run(hook, data)
    assert len(data["messages"]) < len(history) + 1
    assert CTX - tokens(data["messages"]) - MARGIN >= math.ceil(3000 / 1.1)

def test_mandatory_content_over_the_window_is_left_for_vllm_to_reject(hook):
    data = {"model": "chat-model", "messages": [{"role": "user", "content": "z" * (CTX + 10)}], "max_tokens": 3000}
    assert run(hook, data) is None
    assert data["max_tokens"] == 3000


# ── Responses API ───────────────────────────────────────────────────────────

def test_responses_string_input_shrinks_max_output_tokens(hook):
    data = {"model": "chat-model", "input": "z" * 7000, "instructions": "be brief", "max_output_tokens": 3000}
    run(hook, data, "aresponses")
    expected = CTX - tokens([{"role": "system", "content": "be brief"}, {"role": "user", "content": "z" * 7000}]) - MARGIN
    assert data["max_output_tokens"] == expected


def test_responses_without_max_output_tokens_gets_one_when_it_would_overflow(hook):
    data = {"model": "chat-model", "input": "z" * 7000}
    run(hook, data, "aresponses")
    assert data["max_output_tokens"] == CTX - tokens([{"role": "user", "content": "z" * 7000}]) - MARGIN


def test_responses_list_input_drops_oldest_turns(hook):
    items = [m for i in range(6) for m in turn(i, 1000)] + [{"role": "user", "content": "last question"}]
    data = {"model": "chat-model", "input": list(items), "max_output_tokens": 3000}

    assert run(hook, data, "aresponses") is data
    assert data["input"][-1] == items[-1]
    assert len(data["input"]) < len(items)
    assert data["input"][0]["role"] == "user"
    assert data["max_output_tokens"] == 3000


def test_responses_function_call_items_travel_with_their_turn(hook):
    items = [
        {"role": "user", "content": "first"},
        {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "r" * 3000},
        {"role": "assistant", "content": "done"},
        {"role": "user", "content": "q" * 8000},
    ]
    data = {"model": "chat-model", "input": items, "max_output_tokens": 3000}

    run(hook, data, "aresponses")
    assert data["input"] == [items[-1]]


def test_responses_previous_response_history_is_counted(hook):
    stored = [{"role": "user", "content": "h" * 7000}, {"role": "assistant", "content": "ok"}]
    data = {"model": "chat-model", "input": "next", "previous_response_id": "resp_1", "max_output_tokens": 3000}

    with patch("previous_response_hook.load_session_messages", new=AsyncMock(return_value=stored)):
        run(hook, data, "aresponses")

    assert data["max_output_tokens"] == CTX - tokens(stored + [{"role": "user", "content": "next"}]) - MARGIN
    assert data["input"] == "next"


# ── guards kept from before ─────────────────────────────────────────────────

def test_chat_without_user_turn_gets_a_filler_user_message(hook):
    data = {"model": "chat-model", "messages": [{"role": "system", "content": "summarise"}], "max_tokens": 100}
    assert run(hook, data) is data
    assert data["messages"][-1] == {"role": "user", "content": "."}


def test_responses_without_user_turn_gets_a_filler_user_item(hook):
    data = {"model": "chat-model", "instructions": "summarise", "input": [], "max_output_tokens": 100}
    assert run(hook, data, "aresponses") is data
    assert data["input"] == [{"role": "user", "content": "."}]


def test_models_without_a_declared_window_are_untouched(hook):
    data = {"model": "tts-model", "messages": [{"role": "user", "content": "z" * 50_000}], "max_tokens": 3000}
    assert run(hook, data) is None
    assert data["max_tokens"] == 3000


def test_provider_prefix_is_stripped_from_the_model_name(hook):
    data = {"model": "custom_openai/chat-model", "messages": [{"role": "user", "content": "z" * 7000}]}
    assert run(hook, data) is data


def test_other_call_types_are_ignored(hook):
    assert run(hook, {"model": "chat-model", "input": "x" * 50_000}, "aembedding") is None


def test_declared_window_without_a_tokenizer_fails_at_startup(tmp_path):
    bad = {"model_list": [{"model_name": "m", "model_info": {"max_input_tokens": 1000}}]}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="no model_info.tokenizer"):
        ContextTrimHook(config_path=str(path), tokenizer_loader=lambda repo, rev: StubTokenizer())


# ── config.yaml + real tokenizer ────────────────────────────────────────────

def _real_config():
    with open(trim_hook._CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_config_output_limit_matches_litellm_default_for_every_declared_model():
    for entry in _real_config()["model_list"]:
        info = entry.get("model_info") or {}
        if info.get("max_input_tokens"):
            assert info["max_output_tokens"] == entry["litellm_params"]["max_tokens"], entry["model_name"]


def test_config_registers_the_window_in_litellm_model_cost():
    """LiteLLM writes each deployment's model_info into litellm.model_cost."""
    import litellm
    from litellm import Router

    model_list = [e for e in _real_config()["model_list"] if (e.get("model_info") or {}).get("max_input_tokens")]
    Router(model_list=model_list)
    for entry in model_list:
        info = litellm.get_model_info(model=entry["litellm_params"]["model"])
        assert info["max_input_tokens"] == entry["model_info"]["max_input_tokens"]
        assert info["max_output_tokens"] == entry["model_info"]["max_output_tokens"]


def _real_prompt_of(n_tokens, counter):
    """User message of about n_tokens prompt tokens (largest repeat count that
    stays <= n_tokens), plus its exact count."""
    def msg(k):
        return [{"role": "user", "content": "alpha beta gamma delta epsilon zeta eta theta " * k}]

    lo, hi = 1, n_tokens
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(msg(mid), None) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    return msg(lo), counter.count(msg(lo), None)


def test_production_case_98305_prompt_plus_32768_output_is_made_to_fit():
    real = trim_hook.context_trim_hook
    limits = real._limits["qwen3.8-27b"]
    messages, n = _real_prompt_of(98_305, limits.tokenizer)
    assert 97_800 < n <= 98_305   # in the band where input + 32768 overflows 131072

    chat = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": 32768}
    run(real, chat)
    assert n + chat["max_tokens"] <= limits.context_window
    assert chat["messages"] == messages

    responses = {"model": "qwen3.8-27b", "input": messages[0]["content"], "max_output_tokens": 32768}
    run(real, responses, "aresponses")
    assert n + responses["max_output_tokens"] <= limits.context_window
    assert responses["input"] == messages[0]["content"]


def test_real_tokenizer_trims_a_long_conversation_to_fit():
    real = trim_hook.context_trim_hook
    limits = real._limits["qwen3.8-27b"]
    chunk, _ = _real_prompt_of(12_000, limits.tokenizer)
    history = [m for i in range(12) for m in (
        {"role": "user", "content": f"turn {i} " + chunk[0]["content"]},
        {"role": "assistant", "content": "noted"},
    )]
    last = {"role": "user", "content": "final question"}
    data = {"model": "qwen3.8-27b", "messages": history + [last], "max_tokens": 32768}

    assert run(real, data) is data
    kept = data["messages"]
    assert kept[-1] == last and kept[0]["role"] == "user" and len(kept) < len(history) + 1
    n = limits.tokenizer.count(kept, None)
    assert n + data["max_tokens"] + trim_hook._SAFETY_MARGIN <= limits.context_window
    assert data["max_tokens"] == 32768   # trimmed enough that the cap was not needed


def test_real_template_bills_old_thinking_in_full_and_stripping_it_recovers_the_room():
    real = trim_hook.context_trim_hook
    limits = real._limits["qwen3.8-27b"]
    think = "let me think about this carefully step by step " * 1200
    history = [m for i in range(10) for m in (
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}", "reasoning_content": think},
    )]
    last = {"role": "user", "content": "final question"}
    messages = history + [last]
    assert limits.tokenizer.count(messages, None) > 100_000

    data = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": 32768}
    assert run(real, data) is data
    kept = data["messages"]
    assert len(kept) == len(messages)                   # no turn dropped
    assert all("reasoning_content" not in m for m in kept)
    assert limits.tokenizer.count(kept, None) < 2_000
    assert data["max_tokens"] == 32768
