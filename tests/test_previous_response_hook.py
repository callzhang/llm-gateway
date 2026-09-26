import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

import previous_response_hook as mod

_HANDLER = (
    "litellm.responses.litellm_completion_transformation.session_handler."
    "ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id"
)


def _run(data, call_type="aresponses"):
    return asyncio.run(mod.previous_response_history_hook.async_pre_call_hook(None, None, data, call_type))


def test_ignores_other_call_types_and_requests_without_handle():
    assert _run({"input": "x"}) == {"input": "x"}
    assert _run({"previous_response_id": "resp_x"}, call_type="acompletion") == {"previous_response_id": "resp_x"}


def test_passes_when_history_exists():
    with patch(_HANDLER, new=AsyncMock(return_value={"messages": [{"role": "user", "content": "hi"}]})):
        data = {"previous_response_id": "resp_x"}
        assert _run(data) is data


def test_waits_for_history_that_lands_late(monkeypatch):
    monkeypatch.setattr(mod, "_POLL_SECONDS", 0.01)
    handler = AsyncMock(side_effect=[{"messages": []}, {"messages": []}, {"messages": [{"role": "user"}]}])
    with patch(_HANDLER, new=handler):
        _run({"previous_response_id": "resp_x"})
    assert handler.await_count == 3


def test_rejects_when_history_never_appears(monkeypatch):
    monkeypatch.setattr(mod, "_WAIT_SECONDS", 0.05)
    monkeypatch.setattr(mod, "_POLL_SECONDS", 0.01)
    with patch(_HANDLER, new=AsyncMock(return_value={"messages": []})):
        with pytest.raises(HTTPException) as exc:
            _run({"previous_response_id": "resp_x"})
    assert exc.value.status_code == 400
