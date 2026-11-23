"""LiteLLM pre-call hook: the virtual key, not the client, decides who may run at top priority.

Registered in config.yaml:
    litellm_settings:
      callbacks:
        - request_priority_hook.request_priority_hook

model_manager admits queued chat requests by the body field `priority` (0..255, smaller first;
missing = 100; docs/request-priority-gate.md). Clients set it with a nested
`extra_body: {extra_body: {priority: N}}` in the OpenAI SDK, so any key could claim 0. This hook
makes priority 0 a grant held by a key:

  - A key whose metadata has `gateway_priority: P` (an integer 0..255) runs at P when the request
    sends no priority. It may send a larger number (run lower), never a smaller one.
  - Any other key may send 1..255 as before; a 0 it sends becomes PEER_PRIORITY_FLOOR (1).
  - A malformed priority is left alone, so model_manager still answers 400 for it.

Grant a key with LiteLLM's /key/update: `{"key": ..., "metadata": {..., "gateway_priority": 0}}`.
The peer region's requests reach model_manager without passing this hook, so model_manager raises
their priority to the same floor (model_manager.PEER_PRIORITY_FLOOR).
"""
from __future__ import annotations

import logging
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("litellm.request_priority_hook")

MAX_PRIORITY = 255
# The best priority a key without a grant can get; must equal model_manager.PEER_PRIORITY_FLOOR.
UNGRANTED_PRIORITY_FLOOR = 1
# LiteLLM's call types for chat completions and the Responses API; model_manager gates only chat.
CHAT_CALL_TYPES = frozenset({"acompletion", "aresponses"})


def _valid(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_PRIORITY


class RequestPriorityHook(CustomLogger):
    def _grant(self, user_api_key_dict: Any) -> int | None:
        metadata = getattr(user_api_key_dict, "metadata", None) or {}
        if "gateway_priority" not in metadata:
            return None
        grant = metadata["gateway_priority"]
        if _valid(grant):
            return grant
        logger.warning(
            "request_priority_hook: key %s has gateway_priority=%r, not an integer 0..%d; no grant",
            getattr(user_api_key_dict, "key_alias", None), grant, MAX_PRIORITY,
        )
        return None

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict | None:
        if call_type not in CHAT_CALL_TYPES:
            return None
        extra_body = data.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
        grant = self._grant(user_api_key_dict)
        if "priority" in extra_body:
            sent = extra_body["priority"]
            if not _valid(sent):
                return None
            priority = max(sent, UNGRANTED_PRIORITY_FLOOR if grant is None else grant)
        elif grant is not None:
            priority = grant
        else:
            return None
        if extra_body.get("priority") == priority:
            return None
        data["extra_body"] = {**extra_body, "priority": priority}
        return data


# Module-level instance — config.yaml references this via "request_priority_hook.request_priority_hook"
request_priority_hook = RequestPriorityHook()
