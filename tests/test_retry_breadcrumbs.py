"""A spend log must hold only its own request's content (LiteLLM's Router retry breadcrumbs).

LiteLLM 1.99.0 shared one `previous_models` list across requests and copied whole request kwargs
into it, so one caller's spend log stored another's prompt and images (fixed upstream by
BerriAI/litellm PR 39491; docs/spend-log-retry-leak.md).  A real Router with this repo's shape
(an `order: 1` deployment that sheds load, an `order: 2` one that answers) and two callers.  The
request snapshot is built the way the proxy builds it: a shallow copy of the request data, so its
`body.litellm_metadata` / `body.metadata` is the live metadata dict that LiteLLM stores in
LiteLLM_SpendLogs.proxy_server_request.
"""
from __future__ import annotations

import asyncio
import json
import unittest

import litellm
from litellm import Router
from litellm.proxy.spend_tracking.spend_tracking_utils import _sanitize_request_body_for_spend_logs_payload


def router() -> Router:
    def deployment(order: int, mock_response: str) -> dict:
        return {"model_name": "m", "model_info": {"id": f"dep-{order}"}, "litellm_params": {
            "model": "openai/m", "api_key": "x", "api_base": "http://127.0.0.1:9/v1",
            "order": order, "mock_response": mock_response}}

    return Router(
        model_list=[deployment(1, "litellm.InternalServerError"), deployment(2, "ok")],
        num_retries=0, disable_cooldowns=True,
    )


async def call(r: Router, who: str, metadata_key: str) -> dict:
    """One request; returns its snapshot as the spend log would serialise it."""
    image = f"data:image/png;base64,IMAGE-OF-{who}"
    messages = [{"role": "user", "content": [
        {"type": "text", "text": f"SECRET-OF-{who}"}, {"type": "image_url", "image_url": {"url": image}}]}]
    data = {"model": "m", "messages": messages, metadata_key: {"user_api_key_alias": who}}
    data["proxy_server_request"] = {"body": None}
    data["proxy_server_request"]["body"] = {k: v for k, v in data.items() if k != "proxy_server_request"}
    snapshot = data["proxy_server_request"]
    if metadata_key == "metadata":
        await r.acompletion(**data)
    else:
        # The Responses API keeps router metadata under litellm_metadata.
        async def original(**kwargs):
            return await litellm.acompletion(
                **{**kwargs.pop("litellm_params_of_deployment"), "messages": kwargs["messages"]})

        async def pick(**kwargs):
            deployment = await r.async_get_available_deployment(model=kwargs["model"], request_kwargs=kwargs)
            return await original(litellm_params_of_deployment=dict(deployment["litellm_params"]), **kwargs)

        await r.async_function_with_fallbacks(original_function=pick, **data)
    return _sanitize_request_body_for_spend_logs_payload(snapshot["body"])


async def two_callers(metadata_key: str) -> tuple[Router, str, str]:
    r = router()
    first = await call(r, "ALICE", metadata_key)
    second = await call(r, "BOB", metadata_key)
    return r, json.dumps(first, default=str), json.dumps(second, default=str)


class RetryBreadcrumbTests(unittest.TestCase):
    def test_a_log_holds_only_its_own_request(self) -> None:
        for metadata_key in ("metadata", "litellm_metadata"):
            with self.subTest(metadata_key):
                _, alice, bob = asyncio.run(two_callers(metadata_key))
                self.assertNotIn("ALICE", bob)
                self.assertNotIn("BOB", alice)
                breadcrumbs = json.loads(bob)[metadata_key]["previous_models"]
                self.assertEqual(1, len(breadcrumbs))
                self.assertEqual("InternalServerError", breadcrumbs[0]["exception_type"])
                # The content appears once, in the request's own messages.
                self.assertEqual(1, bob.count("SECRET-OF-BOB"))


if __name__ == "__main__":
    unittest.main()
