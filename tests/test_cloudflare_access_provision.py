import unittest

from scripts import provision_cloudflare_access as provision


class PayloadTests(unittest.TestCase):
    def test_organization_payload_is_exact(self):
        self.assertEqual(
            {
                "auth_domain": "stardust-ai.cloudflareaccess.com",
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

    def test_tts_app_has_no_managed_oauth(self):
        # The Skill authenticates with `cloudflared access login`, so dynamic
        # loopback client registration would be an open door with no caller.
        payload = provision.application_payload(
            "Stardust TTS API", "tts-api.preseen.ai", "idp-1"
        )
        self.assertEqual(
            {
                "name": "Stardust TTS API",
                "domain": "tts-api.preseen.ai",
                "type": "self_hosted",
                "session_duration": "24h",
                "auto_redirect_to_identity": True,
                "allowed_idps": ["idp-1"],
            },
            payload,
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
            "Stardust TTS API", "tts-api.preseen.ai", "idp-1"
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


class ErrorDetailTests(unittest.TestCase):
    def test_permission_failures_are_readable(self):
        # The shape Cloudflare actually returns for a token missing a scope.
        body = {"errors": [{"code": 1010, "error": "auth.forbidden"}]}
        self.assertEqual("auth.forbidden (code 1010)", provision._error_detail(body))

    def test_message_field_is_preferred_when_present(self):
        body = {"errors": [{"code": 10000, "message": "Authentication error"}]}
        self.assertEqual(
            "Authentication error (code 10000)", provision._error_detail(body)
        )

    def test_empty_envelope_yields_empty_string(self):
        self.assertEqual("", provision._error_detail({"errors": []}))


class OrganizationDegradeTests(unittest.TestCase):
    """A token scoped to apps only must still be able to provision an app."""

    class _Client:
        def __init__(self, error):
            self.error = error
            self.calls = []

        def request(self, method, path, payload=None):
            self.calls.append((method, path, payload))
            raise self.error

    def test_permission_error_reports_unverified_without_writing(self):
        client = self._Client(provision.CloudflareError("Authentication error"))
        result = provision.reconcile_organization(client, apply=True)
        self.assertEqual("unverified", result["status"])
        # Never PUT or POST a guessed auth_domain: team domains are a global
        # namespace, and the wrong one points the gateway at a foreign JWKS.
        self.assertEqual(["GET"], [call[0] for call in client.calls])

    def test_unrelated_errors_still_propagate(self):
        client = self._Client(provision.CloudflareError("Rate limited"))
        with self.assertRaises(provision.CloudflareError):
            provision.reconcile_organization(client, apply=True)

    def test_unverified_is_not_treated_as_actionable(self):
        self.assertNotIn("unverified", provision.ACTIONABLE_STATUSES)


class SelectorTests(unittest.TestCase):
    """`--only tts` must never touch the Open WebUI hostname.

    Provisioning both applications would put llm.preseen.ai behind Access as a
    side effect, locking out every chat user unless run_open_webui.sh flips to
    trusted-header auth in the same breath.
    """

    def _client(self):
        return FakeClient(
            {
                ("GET", "/access/organizations"): provision.organization_payload(),
                ("GET", "/access/identity_providers"): [
                    {"id": "idp-1", **provision.identity_provider_payload()}
                ],
                ("GET", "/access/apps"): [
                    {
                        "id": "app-tts",
                        "domain": "tts-api.preseen.ai",
                        "aud": "aud-tts",
                        **provision.application_payload(
                            "Stardust TTS API", "tts-api.preseen.ai", "idp-1"
                        ),
                    }
                ],
                ("GET", "/access/apps/app-tts/policies"): [
                    {"id": "p-1", **provision.employee_policy_payload()}
                ],
            }
        )

    def test_only_tts_skips_the_web_application(self):
        client = self._client()
        summaries = provision.provision(client, apply=True, only="tts")
        kinds = [item["kind"] for item in summaries]
        self.assertIn("tts", kinds)
        self.assertNotIn("web", kinds)
        payloads = [call[2] for call in client.calls if call[2]]
        self.assertNotIn(
            "llm.preseen.ai",
            [payload.get("domain") for payload in payloads],
        )

    def test_unknown_selector_is_rejected(self):
        with self.assertRaises(ValueError):
            provision.provision(self._client(), apply=False, only="nope")


if __name__ == "__main__":
    unittest.main()
