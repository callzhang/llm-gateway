"""Fail-closed runtime configuration for the TTS access gateway."""

from __future__ import annotations

import os
from dataclasses import dataclass


def required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


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
        litellm_base_url = os.getenv(
            "TTS_GATEWAY_LITELLM_BASE_URL",
            "http://127.0.0.1:8900",
        ).rstrip("/")
        return cls(
            team_domain=team_domain,
            policy_audience=policy_audience,
            litellm_api_key=litellm_api_key,
            allowed_service_ids=allowed_service_ids,
            litellm_base_url=litellm_base_url,
        )
