import unittest

from scripts import provision_cloudflare_access as provision


class PayloadTests(unittest.TestCase):
    def test_organization_payload_is_exact(self):
        self.assertEqual(
            {
                "auth_domain": "stardust.cloudflareaccess.com",
                "name": "Stardust",
                "auto_redirect_to_identity": True,
                "session_duration": "24h",
            },
            provision.organization_payload(),
        )

    def test_employee_policy_payload_is_exact(self):
        self.assertEqual(
            {
                "name": "Stardust employees",
                "decision": "allow",
                "include": [{"email_domain": {"domain": "stardust.ai"}}],
            },
            provision.employee_policy_payload(),
        )

    def test_tts_app_enables_short_lived_managed_oauth(self):
        payload = provision.application_payload(
            "Stardust TTS API", "tts-api.preseen.ai", "idp-1", managed_oauth=True
        )
        self.assertEqual(
            {
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
            },
            payload["oauth_configuration"],
        )


class FakeClient:
    def __init__(self, resources):
        self.resources = resources
        self.calls = []

    def request(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        return self.resources[(method, path)]


class ReconcileTests(unittest.TestCase):
    def test_matching_application_is_updated_not_duplicated(self):
        client = FakeClient(
            {
                ("GET", "/access/apps"): [
                    {"id": "app-1", "name": "Old", "domain": "tts-api.preseen.ai"}
                ],
                ("PUT", "/access/apps/app-1"): {
                    "id": "app-1",
                    "name": "Stardust TTS API",
                    "domain": "tts-api.preseen.ai",
                    "aud": "aud-1",
                },
            }
        )
        desired = provision.application_payload(
            "Stardust TTS API", "tts-api.preseen.ai", "idp-1", managed_oauth=True
        )

        result = provision.reconcile_named(
            client, "/access/apps", desired, match_key="domain", apply=True
        )

        self.assertEqual("app-1", result["id"])
        self.assertNotIn("POST", [call[0] for call in client.calls])
        self.assertIn(("PUT", "/access/apps/app-1", desired), client.calls)

    def test_matching_policy_is_unchanged_when_payload_matches(self):
        desired = provision.employee_policy_payload()
        client = FakeClient(
            {("GET", "/access/apps/app-1/policies"): [{"id": "p-1", **desired}]}
        )
        result = provision.reconcile_named(
            client,
            "/access/apps/app-1/policies",
            desired,
            match_key="name",
            apply=True,
        )
        self.assertEqual("p-1", result["id"])
        self.assertEqual(1, len(client.calls))


if __name__ == "__main__":
    unittest.main()
