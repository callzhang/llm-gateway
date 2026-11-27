"""Router retry breadcrumbs must describe only their own request (retry_breadcrumb_hook.py).

A real Router with this repo's shape (an `order: 1` deployment that sheds load, an `order: 2`
one that answers) and two callers.  The order fallback calls Router.log_retry even with
num_retries 0, and the request snapshot is built the way the proxy builds it: a shallow copy of
the request data, so its `body.litellm_metadata` / `body.metadata` is the live metadata dict that
LiteLLM stores in LiteLLM_SpendLogs.proxy_server_request.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import unittest
from unittest import mock

import litellm
import yaml
from litellm import Router
from litellm.proxy.spend_tracking.spend_tracking_utils import _sanitize_request_body_for_spend_logs_payload

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM_LOG_RETRY = Router.log_retry


def load_hook():
    """Load by path, as the proxy does.  Loading it patches Router.log_retry."""
    spec = importlib.util.spec_from_file_location("retry_breadcrumb_hook", os.path.join(REPO, "retry_breadcrumb_hook.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    def test_both_regions_register_the_hook(self) -> None:
        for name in ("config.yaml", "config.cn.yaml"):
            with open(os.path.join(REPO, name)) as f:
                callbacks = yaml.safe_load(f)["litellm_settings"]["callbacks"]
            self.assertIn("retry_breadcrumb_hook.retry_breadcrumb_hook", callbacks, name)

    def test_upstream_log_retry_copies_one_callers_content_into_anothers_log(self) -> None:
        """The bug.  If this fails after a LiteLLM upgrade (fixed upstream in 1.101.0, PR 39491),
        delete retry_breadcrumb_hook.py, its callbacks entries and this file."""
        with mock.patch.object(Router, "log_retry", UPSTREAM_LOG_RETRY):
            for metadata_key in ("metadata", "litellm_metadata"):
                with self.subTest(metadata_key):
                    _, _, bob = asyncio.run(two_callers(metadata_key))
                    self.assertIn("SECRET-OF-ALICE", bob)
                    self.assertIn("IMAGE-OF-ALICE", bob)

    def test_with_the_hook_a_log_holds_only_its_own_request(self) -> None:
        with mock.patch.object(Router, "log_retry", UPSTREAM_LOG_RETRY):
            load_hook()
            for metadata_key in ("metadata", "litellm_metadata"):
                with self.subTest(metadata_key):
                    r, alice, bob = asyncio.run(two_callers(metadata_key))
                    self.assertNotIn("ALICE", bob)
                    self.assertNotIn("BOB", alice)
                    self.assertEqual([], r.previous_models)
                    body = json.loads(bob)
                    breadcrumbs = body[metadata_key]["previous_models"]
                    self.assertEqual(1, len(breadcrumbs))
                    self.assertEqual("InternalServerError", breadcrumbs[0]["exception_type"])
                    self.assertEqual(
                        {"model_group", "deployment_id", "exception_type", "exception_string"}, set(breadcrumbs[0]))
                    # The content appears once, in the request's own messages.
                    self.assertEqual(1, bob.count("SECRET-OF-BOB"))

    def test_breadcrumbs_are_capped_bounded_and_never_shared(self) -> None:
        with mock.patch.object(Router, "log_retry", UPSTREAM_LOG_RETRY):
            load_hook()
            r = router()
            kwargs = {"model": "m", "input": "PROMPT", "litellm_metadata": {}}
            for n in range(6):
                held = kwargs["litellm_metadata"].get("previous_models")
                before = list(held or [])
                r.log_retry(kwargs=kwargs, e=ValueError(f"{n}:" + "x" * 5000))
                self.assertEqual(before, list(held or []), "an earlier list must not be mutated")
            crumbs = kwargs["litellm_metadata"]["previous_models"]
            self.assertEqual(["2", "3", "4", "5"], [c["exception_string"].split(":")[0] for c in crumbs])
            self.assertTrue(all(len(c["exception_string"]) <= 500 for c in crumbs))
            self.assertNotIn("PROMPT", json.dumps(crumbs))


if __name__ == "__main__":
    unittest.main()
