from __future__ import annotations

import os
import time
import unittest
from unittest.mock import patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from tts_access_gateway.auth import (
    AccessDenied,
    AccessJWTVerifier,
    principal_from_claims,
)
from tts_access_gateway.config import GatewayConfig


class ConfigTests(unittest.TestCase):
    def test_missing_security_values_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TTS_ACCESS_TEAM_DOMAIN"):
                GatewayConfig.from_env()

    def test_config_normalizes_domain_and_service_allowlist(self):
        with patch.dict(
            os.environ,
            {
                "TTS_ACCESS_TEAM_DOMAIN": (
                    "https://stardust.cloudflareaccess.com/"
                ),
                "TTS_ACCESS_POLICY_AUD": "policy-aud",
                "TTS_GATEWAY_LITELLM_KEY": "internal-key",
                "TTS_ACCESS_SERVICE_CLIENT_IDS": " service-a,service-b,service-a ",
            },
            clear=True,
        ):
            config = GatewayConfig.from_env()

        self.assertEqual(
            "https://stardust.cloudflareaccess.com", config.team_domain
        )
        self.assertEqual(
            frozenset({"service-a", "service-b"}), config.allowed_service_ids
        )


class PrincipalTests(unittest.TestCase):
    def test_employee_requires_exact_company_domain(self):
        principal = principal_from_claims(
            {"sub": "u1", "email": "Person@Stardust.AI", "type": "app"},
            frozenset(),
        )
        self.assertEqual("person@stardust.ai", principal.actor)
        self.assertEqual("employee", principal.kind)

        for email in (
            "person@stardust.ai.example",
            "person@example.com",
            "@stardust.ai",
            "stardust.ai",
        ):
            with self.subTest(email=email):
                with self.assertRaises(AccessDenied):
                    principal_from_claims(
                        {"sub": "u2", "email": email, "type": "app"},
                        frozenset(),
                    )

    def test_service_identity_requires_allowlisted_client_id(self):
        principal = principal_from_claims(
            {"sub": "", "common_name": "svc-id", "type": "app"},
            frozenset({"svc-id"}),
        )
        self.assertEqual("service:svc-id", principal.actor)
        self.assertEqual("service", principal.kind)

        with self.assertRaises(AccessDenied):
            principal_from_claims(
                {"sub": "", "common_name": "unknown", "type": "app"},
                frozenset({"svc-id"}),
            )

    def test_non_application_token_is_rejected(self):
        with self.assertRaises(AccessDenied):
            principal_from_claims(
                {
                    "sub": "u1",
                    "email": "person@stardust.ai",
                    "type": "org",
                },
                frozenset(),
            )


class JwtVerificationTests(unittest.IsolatedAsyncioTestCase):
    async def test_verifies_signature_issuer_audience_and_required_claims(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        now = int(time.time())
        token = jwt.encode(
            {
                "aud": ["policy-aud"],
                "email": "person@stardust.ai",
                "exp": now + 300,
                "iat": now,
                "nbf": now - 1,
                "iss": "https://stardust.cloudflareaccess.com",
                "type": "app",
                "sub": "u1",
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "key-1"},
        )
        verifier = AccessJWTVerifier(
            team_domain="https://stardust.cloudflareaccess.com",
            policy_audience="policy-aud",
            allowed_service_ids=frozenset(),
            signing_key_resolver=lambda _token: private_key.public_key(),
        )

        principal = await verifier.verify(token)

        self.assertEqual("person@stardust.ai", principal.actor)

    async def test_invalid_audience_is_masked(self):
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        now = int(time.time())
        token = jwt.encode(
            {
                "aud": ["wrong-aud"],
                "email": "person@stardust.ai",
                "exp": now + 300,
                "iat": now,
                "nbf": now - 1,
                "iss": "https://stardust.cloudflareaccess.com",
                "type": "app",
                "sub": "u1",
            },
            private_key,
            algorithm="RS256",
            headers={"kid": "key-1"},
        )
        verifier = AccessJWTVerifier(
            team_domain="https://stardust.cloudflareaccess.com",
            policy_audience="policy-aud",
            allowed_service_ids=frozenset(),
            signing_key_resolver=lambda _token: private_key.public_key(),
        )

        with self.assertRaisesRegex(AccessDenied, "^invalid Access JWT$"):
            await verifier.verify(token)


if __name__ == "__main__":
    unittest.main()
