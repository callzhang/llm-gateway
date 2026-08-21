#!/usr/bin/env python3
"""Idempotently provision Stardust Cloudflare Access applications."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen


API_ROOT = "https://api.cloudflare.com/client/v4"


class CloudflareError(RuntimeError):
    pass


# Renamed 2026-08-19 from the auto-generated `late-scene-8b71`.  Note that
# `stardust.cloudflareaccess.com` and `preseen.cloudflareaccess.com` both resolve
# but belong to OTHER organizations — cloudflareaccess.com is a global namespace,
# and trusting the wrong one would point the gateway at a foreign JWKS.  Always
# read this back from the account rather than assuming it.
def organization_payload() -> dict[str, Any]:
    return {
        "auth_domain": "stardust-ai.cloudflareaccess.com",
        "name": "Stardust",
        "auto_redirect_to_identity": True,
        "session_duration": "24h",
    }


def employee_policy_payload() -> dict[str, Any]:
    return {
        "name": "Stardust employees",
        "decision": "allow",
        "include": [{"email_domain": {"domain": "stardust.ai"}}],
    }


def identity_provider_payload() -> dict[str, Any]:
    return {"name": "One-time PIN login", "type": "onetimepin", "config": {}}


# Managed OAuth is enabled only for API applications whose Skills use dynamic
# loopback client registration.  Browser-only applications keep it disabled.
def application_payload(
    name: str,
    domain: str,
    idp_id: str,
    *,
    managed_oauth: bool = False,
    policy_id: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "domain": domain,
        "type": "self_hosted",
        "session_duration": "24h",
        "auto_redirect_to_identity": True,
        "allowed_idps": [idp_id],
    }
    if policy_id:
        payload["policies"] = [{"id": policy_id, "precedence": 1}]
    if managed_oauth:
        payload["oauth_configuration"] = {
            "enabled": True,
            "dynamic_client_registration": {
                "enabled": True,
                "allow_any_on_localhost": False,
                "allow_any_on_loopback": True,
                "allowed_uris": [],
            },
            "grant": {
                "access_token_lifetime": "15m",
                "session_duration": "168h",
            },
        }
    return payload


def _error_detail(body: Any) -> str:
    """Readable text for one Cloudflare error envelope.

    Cloudflare is not consistent about the field: permission failures come back
    as {"code": 1010, "error": "auth.forbidden"} with no "message" at all, which
    a message-only reader renders as the useless "Cloudflare API error".
    """
    parts = []
    for item in (body or {}).get("errors") or []:
        if not isinstance(item, dict):
            parts.append(str(item))
            continue
        text = item.get("message") or item.get("error") or ""
        code = item.get("code")
        if text and code:
            parts.append(f"{text} (code {code})")
        elif text:
            parts.append(str(text))
        elif code:
            parts.append(f"code {code}")
    return "; ".join(parts)


def _contains(actual: Any, desired: Any) -> bool:
    if isinstance(desired, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in desired.items()
        )
    if isinstance(desired, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(desired)
            and all(
                _contains(actual_item, desired_item)
                for actual_item, desired_item in zip(actual, desired, strict=True)
            )
        )
    return actual == desired


class CloudflareClient:
    def __init__(self, account_id: str, token: str):
        self.base = f"{API_ROOT}/accounts/{account_id}"
        self.token = token

    def request(
        self, method: str, path: str, payload: dict[str, Any] | None = None
    ) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            self.base + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urlopen(request, timeout=30) as response:
                body = json.load(response)
        except HTTPError as exc:
            try:
                detail = _error_detail(json.load(exc))
            except Exception:
                detail = ""
            raise CloudflareError(detail or f"HTTP {exc.code}") from exc
        if not body.get("success"):
            raise CloudflareError(
                _error_detail(body) or "Cloudflare API request failed"
            )
        return body.get("result")


def reconcile_named(
    client: Any,
    collection_path: str,
    desired: dict[str, Any],
    *,
    match_key: str,
    apply: bool,
) -> dict[str, Any]:
    resources = client.request("GET", collection_path) or []
    match = next(
        (item for item in resources if item.get(match_key) == desired[match_key]),
        None,
    )
    if match is None:
        if not apply:
            return {**desired, "status": "missing"}
        return client.request("POST", collection_path, desired)
    if _contains(match, desired):
        return match
    if not apply:
        return {**match, "status": "drifted"}
    return client.request("PUT", f"{collection_path}/{match['id']}", desired)


# Statuses that mean a human still has to do something.  "unverified" is
# informational: the resource exists but this token cannot read it back.
ACTIONABLE_STATUSES = {"missing", "drifted"}


def reconcile_organization(client: Any, *, apply: bool) -> dict[str, Any]:
    desired = organization_payload()
    try:
        current = client.request("GET", "/access/organizations")
    except CloudflareError as exc:
        message = str(exc).lower()
        if "not enabled" in message:
            if not apply:
                return {**desired, "status": "missing"}
            return client.request("POST", "/access/organizations", desired)
        if "authentication error" in message or "unauthorized" in message:
            # The token has Access: Apps and Policies but not Access:
            # Organizations, Identity Providers, and Groups.  Provisioning one
            # application does not need to edit the organization, so report it
            # unverified rather than refusing to do the work we can do.
            #
            # Importantly this returns before any write: never PUT a guessed
            # auth_domain.  Team domains are a global namespace on
            # cloudflareaccess.com, so the wrong value would point the gateway
            # at another organization's JWKS — read it from the dashboard, or
            # from the login redirect of an application in this account.
            return {**desired, "status": "unverified"}
        raise
    if _contains(current, desired):
        return current
    if not apply:
        return {**current, "status": "drifted"}
    return client.request("PUT", "/access/organizations", desired)


def _safe_summary(kind: str, resource: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in {
            "kind": kind,
            "id": resource.get("id"),
            "name": resource.get("name"),
            "domain": resource.get("domain") or resource.get("auth_domain"),
            "aud": resource.get("aud"),
            "status": resource.get("status", "ready"),
        }.items()
        if value is not None
    }


APPLICATIONS: dict[str, tuple[str, str, bool]] = {
    "web": ("Stardust LLM Web", "llm.preseen.ai", False),
    "tts": ("Stardust TTS API", "tts-api.preseen.ai", True),
    "ocr": ("Stardust OCR API", "ocr.preseen.ai", True),
    "video_transcribe": (
        "Stardust Video Transcribe API",
        "video-transcribe.preseen.ai",
        True,
    ),
}


def provision(client: Any, *, apply: bool, only: str = "all") -> list[dict[str, Any]]:
    """Reconcile Access state.  ``only`` selects which applications to touch.

    Defaulting to every application would put Open WebUI behind Access as a
    side effect of provisioning TTS — a cutover that must move in lockstep with
    CLOUDFLARE_ACCESS_AUTH_ENABLED in run_open_webui.sh or every chat user is
    locked out.  Callers say which one they mean.
    """
    if only not in {"all", *APPLICATIONS}:
        raise ValueError(f"unknown application selector: {only}")
    output = [
        _safe_summary("organization", reconcile_organization(client, apply=apply))
    ]
    idp = reconcile_named(
        client,
        "/access/identity_providers",
        identity_provider_payload(),
        match_key="name",
        apply=apply,
    )
    output.append(_safe_summary("identity_provider", idp))
    if not idp.get("id"):
        return output

    employee_policy = reconcile_named(
        client,
        "/access/policies",
        employee_policy_payload(),
        match_key="name",
        apply=apply,
    )
    output.append(_safe_summary("employee_policy", employee_policy))
    if not employee_policy.get("id"):
        return output

    selected = APPLICATIONS if only == "all" else {only: APPLICATIONS[only]}
    for kind, (name, domain, managed_oauth) in selected.items():
        app = reconcile_named(
            client,
            "/access/apps",
            application_payload(
                name,
                domain,
                idp["id"],
                managed_oauth=managed_oauth,
                policy_id=employee_policy["id"],
            ),
            match_key="domain",
            apply=apply,
        )
        output.append(_safe_summary(kind, app))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--only",
        choices=("all", *APPLICATIONS),
        default="all",
        help="which Access application to reconcile (default: all)",
    )
    args = parser.parse_args(argv)
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        parser.error("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID are required")
    try:
        summaries = provision(
            CloudflareClient(account_id=account_id, token=token),
            apply=args.apply,
            only=args.only,
        )
    except CloudflareError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True))
    return (
        1 if any(item.get("status") in ACTIONABLE_STATUSES for item in summaries) else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
