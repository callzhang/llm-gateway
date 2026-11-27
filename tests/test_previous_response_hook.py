import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import yaml
from fastapi import HTTPException

import previous_response_hook as mod

_HANDLER = (
    "litellm.responses.litellm_completion_transformation.session_handler."
    "ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id"
)
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_ID = "dfe5c6d1faff58fc"


def _run(data, call_type="aresponses"):
    return asyncio.run(mod.previous_response_history_hook.async_pre_call_hook(None, None, data, call_type))


def _managed_id(model_id=_MODEL_ID, response_id="chatcmpl-1"):
    from litellm.responses.utils import ResponsesAPIRequestUtils

    return ResponsesAPIRequestUtils._build_responses_api_response_id("custom_openai", model_id, response_id)


def _encrypted(inner):
    """An id as this proxy hands it out: ResponsesIDSecurity encrypts it with the master key."""
    from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_value_helper
    from litellm.types.utils import SpecialEnums

    value = SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(inner, "", "")
    return "resp_" + encrypt_value_helper(value=value)


@pytest.fixture(autouse=True)
def proxy(monkeypatch):
    """The proxy globals the hook reads: the master key (id encryption) and the router's deployments."""
    import litellm.proxy.proxy_server as proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master")
    router = SimpleNamespace(get_deployment=lambda model_id: object() if model_id == _MODEL_ID else None)
    monkeypatch.setattr(proxy_server, "llm_router", router)


def test_ignores_other_call_types_and_requests_without_handle():
    assert _run({"input": "x"}) == {"input": "x"}
    assert _run({"previous_response_id": "resp_x"}, call_type="acompletion") == {"previous_response_id": "resp_x"}


@pytest.mark.parametrize("make_id", [
    lambda: "resp_bogus_does_not_exist_123",
    lambda: "resp_17f18078-1077-9f17-870c-8aaaaaaaaaaa",          # a bare inner id, never handed out
    lambda: _encrypted("resp_eaeb90a2-aece-458d-be67-d0cb3acac0a5"),  # response.created's placeholder id
    lambda: "not-even-resp",
])
def test_an_id_this_gateway_never_issued_is_410_at_once_without_a_lookup(make_id):
    previous_response_id = make_id()
    handler = AsyncMock(return_value={"messages": [{"role": "user"}]})
    with patch(_HANDLER, new=handler):
        with pytest.raises(HTTPException) as exc:
            _run({"previous_response_id": previous_response_id})
    assert exc.value.status_code == 410
    assert handler.await_count == 0


def test_an_id_naming_another_proxys_deployment_is_410_at_once():
    handler = AsyncMock(return_value={"messages": [{"role": "user"}]})
    with patch(_HANDLER, new=handler):
        with pytest.raises(HTTPException) as exc:
            _run({"previous_response_id": _encrypted(_managed_id(model_id="somewhere-else"))})
    assert exc.value.status_code == 410
    assert handler.await_count == 0


@pytest.mark.parametrize("make_id", [
    lambda: _encrypted(_managed_id()),  # as the client received it
    lambda: _managed_id(),              # after LiteLLM's ResponsesIDSecurity has decrypted it
])
def test_passes_when_history_exists(make_id):
    previous_response_id = make_id()
    with patch(_HANDLER, new=AsyncMock(return_value={"messages": [{"role": "user", "content": "hi"}]})):
        data = {"previous_response_id": previous_response_id}
        assert _run(data) is data


def test_waits_for_history_that_lands_late(monkeypatch):
    monkeypatch.setattr(mod, "_POLL_SECONDS", 0.01)
    handler = AsyncMock(side_effect=[{"messages": []}, {"messages": []}, {"messages": [{"role": "user"}]}])
    with patch(_HANDLER, new=handler):
        _run({"previous_response_id": _managed_id()})
    assert handler.await_count == 3


def test_rejects_when_history_never_appears(monkeypatch):
    monkeypatch.setattr(mod, "_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(mod, "_POLL_SECONDS", 0.01)
    with patch(_HANDLER, new=AsyncMock(return_value={"messages": []})):
        with pytest.raises(HTTPException) as exc:
            _run({"previous_response_id": _managed_id()})
    assert exc.value.status_code == 410


def test_the_wait_ends_before_the_first_keepalive_can_commit_a_200():
    """Once LiteLLM sends its first `: ping` (sse_keepalive_ping_interval_seconds) the status line
    is on the wire, and the 410 would reach the client as a generic in-stream error. The last
    lookup starts at the deadline, so leave it a poll interval to finish."""
    cfg = yaml.safe_load(open(os.path.join(_REPO, "config.yaml")))
    interval = cfg["litellm_settings"]["sse_keepalive_ping_interval_seconds"]
    assert mod._WAIT_SECONDS + 2 * mod._POLL_SECONDS <= interval


@pytest.mark.parametrize("name", ["config.yaml", "config.cn.yaml"])
def test_history_is_written_inside_the_wait_in_every_region(name):
    """Spend-log rows (the history) are written every proxy_batch_write_at + 0..5 s (default 10), so a
    follow-up sent right after its response must find its row before the hook answers 410."""
    cfg = yaml.safe_load(open(os.path.join(_REPO, name)))
    assert cfg["general_settings"]["proxy_batch_write_at"] + 5 + mod._POLL_SECONDS <= mod._WAIT_SECONDS
