"""Cloudflare Access JWT verification and principal mapping."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Callable, Mapping

import jwt


class AccessDenied(Exception):
    """Authentication succeeded nowhere or did not meet company policy."""


@dataclass(frozen=True)
class AccessPrincipal:
    kind: str
    actor: str
    subject: str


def principal_from_claims(
    claims: Mapping[str, object],
    allowed_service_ids: frozenset[str],
) -> AccessPrincipal:
    if claims.get("type") != "app":
        raise AccessDenied("application token required")

    email = str(claims.get("email") or "").strip().lower()
    subject = str(claims.get("sub") or "").strip()
    if subject and email:
        local, separator, domain = email.rpartition("@")
        if not separator or not local or domain != "stardust.ai":
            raise AccessDenied("company email required")
        return AccessPrincipal(
            kind="employee",
            actor=email,
            subject=subject,
        )

    common_name = str(claims.get("common_name") or "").strip()
    if common_name and common_name in allowed_service_ids:
        return AccessPrincipal(
            kind="service",
            actor=f"service:{common_name}",
            subject=common_name,
        )
    raise AccessDenied("unapproved Access identity")


class AccessJWTVerifier:
    def __init__(
        self,
        *,
        team_domain: str,
        policy_audience: str,
        allowed_service_ids: frozenset[str],
        signing_key_resolver: Callable[[str], object] | None = None,
    ) -> None:
        self._team_domain = team_domain.rstrip("/")
        self._policy_audience = policy_audience
        self._allowed_service_ids = allowed_service_ids
        if signing_key_resolver is None:
            jwks_client = jwt.PyJWKClient(
                f"{self._team_domain}/cdn-cgi/access/certs",
                cache_jwk_set=True,
                lifespan=3600,
                cache_keys=True,
                timeout=10,
            )
            signing_key_resolver = jwks_client.get_signing_key_from_jwt
        self._signing_key_resolver = signing_key_resolver

    async def verify(self, token: str) -> AccessPrincipal:
        if not token:
            raise AccessDenied("missing Access JWT")
        try:
            resolved = await asyncio.to_thread(
                self._signing_key_resolver,
                token,
            )
            key = getattr(resolved, "key", resolved)
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self._policy_audience,
                issuer=self._team_domain,
                options={
                    "require": [
                        "aud",
                        "exp",
                        "iat",
                        "nbf",
                        "iss",
                        "type",
                    ]
                },
            )
        except (jwt.PyJWTError, OSError, ValueError, TypeError) as exc:
            raise AccessDenied("invalid Access JWT") from exc
        return principal_from_claims(
            claims,
            self._allowed_service_ids,
        )
