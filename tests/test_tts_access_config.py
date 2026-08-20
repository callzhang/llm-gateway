from __future__ import annotations

import unittest
from unittest.mock import patch

from tts_access_gateway.config import GatewayConfig


REQUIRED = {
    "TTS_ACCESS_TEAM_DOMAIN": "https://stardust.cloudflareaccess.com",
    "TTS_ACCESS_POLICY_AUD": "policy-aud",
    "TTS_GATEWAY_LITELLM_KEY": "internal-tts-key",
}


def _env(**extra: str) -> dict[str, str]:
    return {**REQUIRED, **extra}


class ConfigTests(unittest.TestCase):
    def test_defaults_match_the_loopback_deployment(self):
        with patch.dict("os.environ", _env(), clear=True):
            config = GatewayConfig.from_env()
        self.assertEqual("127.0.0.1", config.listen_host)
        self.assertEqual(8910, config.listen_port)
        self.assertEqual("http://127.0.0.1:8900", config.litellm_base_url)

    def test_launcher_variables_are_honoured(self):
        # run_tts_access_gateway.sh exports exactly these names.  They used to
        # be ignored, so changing the port during deployment would have moved
        # nothing while appearing to work.
        env = _env(
            TTS_GATEWAY_HOST="127.0.0.2",
            TTS_GATEWAY_PORT="8911",
            TTS_GATEWAY_UPSTREAM_BASE_URL="http://127.0.0.1:8901/",
        )
        with patch.dict("os.environ", env, clear=True):
            config = GatewayConfig.from_env()
        self.assertEqual("127.0.0.2", config.listen_host)
        self.assertEqual(8911, config.listen_port)
        self.assertEqual("http://127.0.0.1:8901", config.litellm_base_url)

    def test_deprecated_upstream_name_still_applies(self):
        env = _env(TTS_GATEWAY_LITELLM_BASE_URL="http://127.0.0.1:8902")
        with patch.dict("os.environ", env, clear=True):
            config = GatewayConfig.from_env()
        self.assertEqual("http://127.0.0.1:8902", config.litellm_base_url)

    def test_new_upstream_name_wins_over_deprecated(self):
        env = _env(
            TTS_GATEWAY_UPSTREAM_BASE_URL="http://127.0.0.1:8901",
            TTS_GATEWAY_LITELLM_BASE_URL="http://127.0.0.1:8902",
        )
        with patch.dict("os.environ", env, clear=True):
            config = GatewayConfig.from_env()
        self.assertEqual("http://127.0.0.1:8901", config.litellm_base_url)

    def test_missing_configuration_fails_closed(self):
        for missing in REQUIRED:
            env = {key: value for key, value in REQUIRED.items() if key != missing}
            with self.subTest(missing=missing):
                with patch.dict("os.environ", env, clear=True):
                    with self.assertRaises(ValueError):
                        GatewayConfig.from_env()

    def test_plaintext_team_domain_is_rejected(self):
        env = _env(TTS_ACCESS_TEAM_DOMAIN="http://stardust.cloudflareaccess.com")
        with patch.dict("os.environ", env, clear=True):
            with self.assertRaises(ValueError):
                GatewayConfig.from_env()

    def test_service_client_ids_are_split_and_trimmed(self):
        env = _env(TTS_ACCESS_SERVICE_CLIENT_IDS=" a.access , b.access ,, ")
        with patch.dict("os.environ", env, clear=True):
            config = GatewayConfig.from_env()
        self.assertEqual(frozenset({"a.access", "b.access"}), config.allowed_service_ids)


if __name__ == "__main__":
    unittest.main()
