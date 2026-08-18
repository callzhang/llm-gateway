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


def organization_payload() -> dict[str, Any]:
    return {
        "auth_domain": "stardust.cloudflareaccess.com",
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


def application_payload(
    name: str, domain: str, idp_id: str, *, managed_oauth: bool = False
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "domain": domain,
        "type": "self_hosted",
        "session_duration": "24h",
        "auto_redirect_to_identity": True,
        "allowed_idps": [idp_id],
    }
    if managed_oauth:
        payload["oauth_configuration"] = {
            "enabled": True,
            "dynamic_client_registration": {
                "enabled": True,
                "allow_any_on_localhost": True,
                "allow_any_on_loopback": True,
                "allowed_uris": [],
            },
            "grant": {
                "access_token_lifetime": "15m",
                "session_duration": "168h",
            },
        }
    return payload


def _contains(actual: Any, desired: Any) -> bool:
    if isinstance(desired, dict):
        return isinstance(actual, dict) and all(
            key in actual and _contains(actual[key], value)
            for key, value in desired.items()
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
                body = json.load(exc)
                errors = body.get("errors", [])
                detail = "; ".join(
                    str(item.get("message", "Cloudflare API error"))
                    for item in errors
                )
            except Exception:
                detail = f"HTTP {exc.code}"
            raise CloudflareError(detail or f"HTTP {exc.code}") from exc
        if not body.get("success"):
            detail = "; ".join(
                str(item.get("message", "Cloudflare API error"))
                for item in body.get("errors", [])
            )
            raise CloudflareError(detail or "Cloudflare API request failed")
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


def reconcile_organization(client: Any, *, apply: bool) -> dict[str, Any]:
    desired = organization_payload()
    try:
        current = client.request("GET", "/access/organizations")
    except CloudflareError as exc:
        if "not enabled" not in str(exc).lower():
            raise
        if not apply:
            return {**desired, "status": "missing"}
        return client.request("POST", "/access/organizations", desired)
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


def provision(client: Any, *, apply: bool) -> list[dict[str, Any]]:
    output = [_safe_summary("organization", reconcile_organization(client, apply=apply))]
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

    applications = (
        ("web", "Stardust LLM Web", "llm.preseen.ai", False),
        ("tts", "Stardust TTS API", "tts-api.preseen.ai", True),
    )
    for kind, name, domain, managed_oauth in applications:
        app = reconcile_named(
            client,
            "/access/apps",
            application_payload(name, domain, idp["id"], managed_oauth=managed_oauth),
            match_key="domain",
            apply=apply,
        )
        output.append(_safe_summary(kind, app))
        if app.get("id"):
            policy = reconcile_named(
                client,
                f"/access/apps/{app['id']}/policies",
                employee_policy_payload(),
                match_key="name",
                apply=apply,
            )
            output.append(_safe_summary(f"{kind}_policy", policy))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account_id:
        parser.error("CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID are required")
    try:
        summaries = provision(
            CloudflareClient(account_id=account_id, token=token), apply=args.apply
        )
    except CloudflareError as exc:
        print(json.dumps({"status": "error", "message": str(exc)}), file=sys.stderr)
        return 1
    for summary in summaries:
        print(json.dumps(summary, sort_keys=True))
    return 1 if any(item.get("status") != "ready" for item in summaries) else 0


if __name__ == "__main__":
    raise SystemExit(main())
