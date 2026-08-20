"""Fail-closed runtime configuration for the TTS access gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _optional_env(*names: str) -> str | None:
    """First non-empty value among ``names``.

    Several names exist only so a renamed variable keeps working; the launcher
    exports the first name in each group.
    """
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


@dataclass(frozen=True)
class GatewayConfig:
    team_domain: str
    policy_audience: str
    litellm_api_key: str
    allowed_service_ids: frozenset[str]
    litellm_base_url: str = "http://127.0.0.1:8900"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8910

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        team_domain = required_env("TTS_ACCESS_TEAM_DOMAIN").rstrip("/")
        if not team_domain.startswith("https://"):
            raise ValueError("TTS_ACCESS_TEAM_DOMAIN must use https://")
        policy_audience = required_env("TTS_ACCESS_POLICY_AUD")
        litellm_api_key = required_env("TTS_GATEWAY_LITELLM_KEY")
        allowed_service_ids = frozenset(
            value.strip()
            for value in os.getenv(
                "TTS_ACCESS_SERVICE_CLIENT_IDS", ""
            ).split(",")
            if value.strip()
        )
        # TTS_GATEWAY_LITELLM_BASE_URL is the pre-2026-08-19 name, kept so an
        # existing environment file does not silently lose its override.
        litellm_base_url = (
            _optional_env(
                "TTS_GATEWAY_UPSTREAM_BASE_URL",
                "TTS_GATEWAY_LITELLM_BASE_URL",
            )
            or "http://127.0.0.1:8900"
        ).rstrip("/")
        listen_host = _optional_env("TTS_GATEWAY_HOST") or "127.0.0.1"
        listen_port = int(_optional_env("TTS_GATEWAY_PORT") or "8910")
        return cls(
            team_domain=team_domain,
            policy_audience=policy_audience,
            litellm_api_key=litellm_api_key,
            allowed_service_ids=allowed_service_ids,
            litellm_base_url=litellm_base_url,
            listen_host=listen_host,
            listen_port=listen_port,
        )
