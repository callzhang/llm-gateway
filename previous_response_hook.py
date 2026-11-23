"""LiteLLM pre-call hook: reject previous_response_id that has no stored history.

For custom_openai models LiteLLM rebuilds the conversation from
LiteLLM_SpendLogs (proxy_server_request + response).  When that lookup comes
back empty — prompt storage off, id not decodable, row past
maximum_spend_logs_retention_period — LiteLLM silently treats the request as a
fresh one, so the caller believes it has continuity and does not.  This hook
turns that silent no-op into a 410 (Gone) so clients can key on the status instead of the message text.

It reuses LiteLLM's own reconstruction, so "history exists" here means exactly
what the bridge will later see.

It must answer before LiteLLM's first SSE keepalive (config.yaml,
sse_keepalive_ping_interval_seconds): pre-call hooks run inside that wrapper, so
once a `: ping` is sent the 200 is on the wire and a 410 can only reach a
streaming client as a generic in-stream error.  Hence:

  - An id this proxy could never have issued is 410 at once, without a lookup.
    The proxy hands out `resp_<encrypted>` (LiteLLM's ResponsesIDSecurity, with the
    master key), which decrypts to LiteLLM's managed id naming one of this proxy's
    deployments (`model_id`).  Anything else is not ours: a made-up id, a bare
    inner id, another proxy's id, or the placeholder id of `response.created` /
    `response.in_progress` (it decrypts to an id with no deployment).
  - For its own ids it waits for the spend-log row, which is written in batches
    (general_settings.proxy_batch_write_at + 0-5 s), at most
    PREVIOUS_RESPONSE_WAIT_SECONDS, kept below the keepalive interval
    (tests/test_previous_response_hook.py checks both against config.yaml).
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger


_WAIT_SECONDS = float(os.environ.get("PREVIOUS_RESPONSE_WAIT_SECONDS", "12"))
_POLL_SECONDS = 1.0


def _inner_response_id(previous_response_id: str) -> str:
    """The managed id inside what the client sent.  Hook order against LiteLLM's own
    ResponsesIDSecurity is not fixed, so accept the encrypted id and the decrypted one."""
    from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity

    security = ResponsesIDSecurity()
    if security._is_encrypted_response_id(previous_response_id):
        previous_response_id, _, _ = security._decrypt_response_id(previous_response_id)
    return previous_response_id


def issued_here(previous_response_id: str) -> bool:
    """True when the id names one of this proxy's deployments, i.e. this proxy issued it."""
    from litellm.proxy import proxy_server
    from litellm.responses.utils import ResponsesAPIRequestUtils

    decoded = ResponsesAPIRequestUtils._decode_responses_api_response_id(_inner_response_id(previous_response_id))
    model_id = decoded.get("model_id")
    router = proxy_server.llm_router
    return bool(model_id) and router is not None and router.get_deployment(model_id=model_id) is not None


async def load_session_messages(previous_response_id: str) -> list[dict]:
    """Stored chat messages for a previous_response_id, exactly as the bridge
    will rebuild them.  Empty when nothing is stored."""
    from litellm.responses.litellm_completion_transformation.session_handler import (
        ResponsesSessionHandler,
    )

    session = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
        previous_response_id=_inner_response_id(previous_response_id)
    )
    return session.get("messages") or []


class PreviousResponseHistoryHook(CustomLogger):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        if call_type != "aresponses":
            return data
        previous_response_id = data.get("previous_response_id")
        if not previous_response_id:
            return data
        if not issued_here(str(previous_response_id)):
            raise HTTPException(
                status_code=410,
                detail=(
                    "previous_response_id was not issued by this gateway. Continue from the id of "
                    "the finished response (response.completed, or the non-streaming response), "
                    "or send a request without previous_response_id to start a new thread."
                ),
            )

        deadline = time.monotonic() + _WAIT_SECONDS
        while True:
            messages = await load_session_messages(previous_response_id)
            if messages or time.monotonic() >= deadline:
                break
            await asyncio.sleep(_POLL_SECONDS)
        if not messages:
            raise HTTPException(
                status_code=410,
                detail=(
                    "previous_response_id has no stored history: the response is unknown, "
                    "expired, or store_prompts_in_spend_logs is off. Send a request "
                    "without previous_response_id to start a new thread."
                ),
            )
        return data


previous_response_history_hook = PreviousResponseHistoryHook()
