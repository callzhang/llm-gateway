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

    def test_browser_app_has_no_managed_oauth(self):
        # Browser-only applications do not need dynamic loopback registration.
        payload = provision.application_payload(
            "Stardust LLM Web", "llm.preseen.ai", "idp-1"
        )
        self.assertEqual(
            {
                "name": "Stardust LLM Web",
                "domain": "llm.preseen.ai",
                "type": "self_hosted",
                "session_duration": "24h",
                "auto_redirect_to_identity": True,
                "allowed_idps": ["idp-1"],
            },
            payload,
        )

    def test_gpu_api_apps_enable_short_lived_managed_oauth(self):
        expected = {
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

        for kind in ("tts", "ocr", "video_transcribe"):
            name, domain, managed_oauth = provision.APPLICATIONS[kind]
            with self.subTest(kind=kind):
                payload = provision.application_payload(
                    name,
                    domain,
                    "idp-1",
                    managed_oauth=managed_oauth,
                    policy_id="employee-policy-1",
                )
                self.assertEqual(expected, payload["oauth_configuration"])
                self.assertEqual(
                    [{"id": "employee-policy-1", "precedence": 1}],
                    payload["policies"],
                )

    def test_application_catalog_keeps_gpu_services_separate(self):
        self.assertEqual(
            {
                "web": ("Stardust LLM Web", "llm.preseen.ai", False),
                "tts": ("Stardust TTS API", "tts-api.preseen.ai", True),
                "ocr": ("Stardust OCR API", "ocr.preseen.ai", True),
                "video_transcribe": (
                    "Stardust Video Transcribe API",
                    "video-transcribe.preseen.ai",
                    True,
                ),
            },
            provision.APPLICATIONS,
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

    def test_expanded_reusable_policy_link_does_not_cause_false_drift(self):
        desired = provision.application_payload(
            "Stardust OCR API",
            "ocr.preseen.ai",
            "idp-1",
            managed_oauth=True,
            policy_id="employee-policy-1",
        )
        current = {
            "id": "app-ocr",
            **desired,
            "policies": [
                {
                    "id": "employee-policy-1",
                    "precedence": 1,
                    "name": "Stardust employees",
                    "decision": "allow",
                }
            ],
        }
        client = FakeClient({("GET", "/access/apps"): [current]})

        result = provision.reconcile_named(
            client, "/access/apps", desired, match_key="domain", apply=True
        )

        self.assertEqual("app-ocr", result["id"])
        self.assertEqual([("GET", "/access/apps", None)], client.calls)


class ReusablePolicyTests(unittest.TestCase):
    def test_selected_app_reuses_account_policy_without_creating_legacy_copy(self):
        desired_policy = provision.employee_policy_payload()
        client = FakeClient(
            {
                ("GET", "/access/organizations"): provision.organization_payload(),
                ("GET", "/access/identity_providers"): [
                    {"id": "idp-1", **provision.identity_provider_payload()}
                ],
                ("GET", "/access/policies"): [
                    {"id": "employee-policy-1", **desired_policy}
                ],
                ("GET", "/access/apps"): [],
                ("POST", "/access/apps"): {
                    "id": "app-ocr",
                    "aud": "aud-ocr",
                    "name": "Stardust OCR API",
                    "domain": "ocr.preseen.ai",
                },
            }
        )

        summaries = provision.provision(client, apply=True, only="ocr")

        self.assertIn("employee_policy", [item["kind"] for item in summaries])
        app_write = next(
            call for call in client.calls if call[:2] == ("POST", "/access/apps")
        )
        self.assertEqual(
            [{"id": "employee-policy-1", "precedence": 1}],
            app_write[2]["policies"],
        )
        self.assertNotIn(
            "/access/apps/app-ocr/policies", [call[1] for call in client.calls]
        )


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
        policy_id = "employee-policy-1"
        return FakeClient(
            {
                ("GET", "/access/organizations"): provision.organization_payload(),
                ("GET", "/access/identity_providers"): [
                    {"id": "idp-1", **provision.identity_provider_payload()}
                ],
                ("GET", "/access/policies"): [
                    {"id": policy_id, **provision.employee_policy_payload()}
                ],
                ("GET", "/access/apps"): [
                    {
                        "id": "app-tts",
                        "domain": "tts-api.preseen.ai",
                        "aud": "aud-tts",
                        **provision.application_payload(
                            "Stardust TTS API",
                            "tts-api.preseen.ai",
                            "idp-1",
                            managed_oauth=True,
                            policy_id=policy_id,
                        ),
                    }
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
