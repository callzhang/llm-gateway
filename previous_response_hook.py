"""LiteLLM pre-call hook: reject previous_response_id that has no stored history.

For custom_openai models LiteLLM rebuilds the conversation from
LiteLLM_SpendLogs (proxy_server_request + response).  When that lookup comes
back empty — prompt storage off, id not decodable, row past
maximum_spend_logs_retention_period — LiteLLM silently treats the request as a
fresh one, so the caller believes it has continuity and does not.  This hook
turns that silent no-op into a 410 (Gone) so clients can key on the status instead of the message text.

It reuses LiteLLM's own reconstruction, so "history exists" here means exactly
what the bridge will later see.

Spend-log rows are written in batches, so the previous response becomes visible
7-12 s after it was returned (measured on LiteLLM 1.99.0).  A follow-up sent
inside that window would silently lose its history, so the hook waits up to
PREVIOUS_RESPONSE_WAIT_SECONDS for the row before answering 410.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from fastapi import HTTPException
from litellm.integrations.custom_logger import CustomLogger


_WAIT_SECONDS = float(os.environ.get("PREVIOUS_RESPONSE_WAIT_SECONDS", "30"))
_POLL_SECONDS = 1.0


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

        from litellm.proxy.hooks.responses_id_security import ResponsesIDSecurity
        from litellm.responses.litellm_completion_transformation.session_handler import (
            ResponsesSessionHandler,
        )

        # Hook order against LiteLLM's own ResponsesIDSecurity is not fixed, so
        # accept both the encrypted id and the already-decrypted one.
        security = ResponsesIDSecurity()
        if security._is_encrypted_response_id(previous_response_id):
            previous_response_id, _, _ = security._decrypt_response_id(previous_response_id)

        deadline = time.monotonic() + _WAIT_SECONDS
        while True:
            session = await ResponsesSessionHandler.get_chat_completion_message_history_for_previous_response_id(
                previous_response_id=previous_response_id
            )
            if session.get("messages") or time.monotonic() >= deadline:
                break
            await asyncio.sleep(_POLL_SECONDS)
        if not session.get("messages"):
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
