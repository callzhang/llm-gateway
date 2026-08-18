# Cloudflare Access TTS Authentication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace employee-distributed LiteLLM keys with Cloudflare Access email OTP and Managed OAuth while keeping all LiteLLM credentials on GPU4.

**Architecture:** Cloudflare Access protects Open WebUI and a new TTS-only hostname. A loopback aiohttp gateway validates Cloudflare application JWTs, enforces TTS routes and payloads, and proxies with a server-side LiteLLM virtual key. The public Skill performs Managed OAuth PKCE and stores refresh credentials only in macOS Keychain.

**Tech Stack:** Python 3.12, aiohttp, PyJWT with cryptography, unittest, systemd user services, Cloudflare Tunnel and Access REST APIs, macOS Keychain, LiteLLM 1.86.2, Open WebUI 0.9.6.

---

## File map

### llm-gateway

- Create `tts_access_gateway/__init__.py`: package marker and version.
- Create `tts_access_gateway/config.py`: fail-closed environment parsing.
- Create `tts_access_gateway/auth.py`: Cloudflare JWKS verification and principal mapping.
- Create `tts_access_gateway/policy.py`: route, model, voice, text, and response-format policy.
- Create `tts_access_gateway/app.py`: aiohttp application, audit logging, and upstream streaming.
- Create `tts_access_gateway/__main__.py`: process entrypoint.
- Create `requirements-tts-access.txt`: isolated pinned runtime dependencies.
- Create `run_tts_access_gateway.sh`: environment checks and foreground runner.
- Create `systemd/llm-tts-access-gateway.service`: loopback gateway supervision.
- Create `tests/test_tts_access_auth.py`: JWT and identity tests.
- Create `tests/test_tts_access_policy.py`: route and payload policy tests.
- Create `tests/test_tts_access_app.py`: HTTP proxy and header-stripping tests.
- Create `tests/test_cloudflare_access_provision.py`: deterministic Cloudflare payload tests.
- Create `scripts/provision_cloudflare_access.py`: idempotent Zero Trust, OTP, app, and policy provisioning.
- Modify `run_open_webui.sh`: trusted-header login and fail-closed settings.
- Modify `systemd/llm-gateway.target`: include the auth gateway.
- Modify `systemd/install.sh`: install the new unit.
- Modify `README.md`: operations, deployment, audit, and rollback.

### stardust-skills

- Create `skills/stardust-tts/scripts/access_oauth.py`: RFC 9728 discovery, dynamic registration, PKCE, callback, refresh, and Keychain storage.
- Modify `skills/stardust-tts/scripts/synthesize.py`: OAuth/service-token authentication and the new base URL.
- Create `skills/stardust-tts/tests/test_access_oauth.py`: OAuth and secure-storage tests.
- Modify `skills/stardust-tts/tests/test_synthesize.py`: remove static-key assumptions and assert Access headers.
- Modify `skills/stardust-tts/SKILL.md` and `README.md`: employee login, headless mode, logout, and security boundaries.
- Modify repository `README.md`: credential table and TTS example.

### GPU4 local infrastructure

- Modify `/home/stardust/preseen-ai-gateway/ingress.json`: add protected `tts-api.preseen.ai` and protect `llm.preseen.ai` with Access audience validation.
- Preserve the repository's existing unrelated dirty changes.

---

### Task 1: Fail-closed gateway configuration and JWT verification

**Files:**
- Create: `tts_access_gateway/__init__.py`
- Create: `tts_access_gateway/config.py`
- Create: `tts_access_gateway/auth.py`
- Test: `tests/test_tts_access_auth.py`

- [ ] **Step 1: Write failing configuration and identity tests**

```python
class ConfigTests(unittest.TestCase):
    def test_missing_security_values_fail_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TTS_ACCESS_TEAM_DOMAIN"):
                GatewayConfig.from_env()

class PrincipalTests(unittest.TestCase):
    def test_employee_requires_exact_company_domain(self):
        principal = principal_from_claims(
            {"sub": "u1", "email": "person@stardust.ai", "type": "app"},
            frozenset(),
        )
        self.assertEqual("person@stardust.ai", principal.actor)
        with self.assertRaises(AccessDenied):
            principal_from_claims(
                {"sub": "u2", "email": "person@stardust.ai.example", "type": "app"},
                frozenset(),
            )

    def test_service_identity_requires_allowlisted_client_id(self):
        principal = principal_from_claims(
            {"sub": "", "common_name": "svc-id", "type": "app"},
            frozenset({"svc-id"}),
        )
        self.assertEqual("service:svc-id", principal.actor)
        with self.assertRaises(AccessDenied):
            principal_from_claims(
                {"sub": "", "common_name": "unknown", "type": "app"},
                frozenset({"svc-id"}),
            )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_tts_access_auth -v
```

Expected: import failure because `tts_access_gateway.config` and
`tts_access_gateway.auth` do not exist.

- [ ] **Step 3: Implement minimal configuration and principal mapping**

```python
@dataclass(frozen=True)
class GatewayConfig:
    team_domain: str
    policy_audience: str
    litellm_api_key: str
    allowed_service_ids: frozenset[str]
    litellm_base_url: str = "http://127.0.0.1:8900"

    @classmethod
    def from_env(cls) -> "GatewayConfig":
        team_domain = required_env("TTS_ACCESS_TEAM_DOMAIN").rstrip("/")
        audience = required_env("TTS_ACCESS_POLICY_AUD")
        key = required_env("TTS_GATEWAY_LITELLM_KEY")
        service_ids = frozenset(
            value.strip()
            for value in os.getenv("TTS_ACCESS_SERVICE_CLIENT_IDS", "").split(",")
            if value.strip()
        )
        return cls(team_domain, audience, key, service_ids)
```

```python
def principal_from_claims(
    claims: Mapping[str, object],
    allowed_service_ids: frozenset[str],
) -> AccessPrincipal:
    email = str(claims.get("email") or "").strip().lower()
    subject = str(claims.get("sub") or "").strip()
    if subject and email:
        local, separator, domain = email.rpartition("@")
        if not separator or not local or domain != "stardust.ai":
            raise AccessDenied("company email required")
        return AccessPrincipal(kind="employee", actor=email, subject=subject)
    common_name = str(claims.get("common_name") or "").strip()
    if common_name and common_name in allowed_service_ids:
        return AccessPrincipal(
            kind="service",
            actor=f"service:{common_name}",
            subject=common_name,
        )
    raise AccessDenied("unapproved Access identity")
```

- [ ] **Step 4: Add RS256 verification with issuer and audience checks**

Use `jwt.PyJWKClient` with
`config.team_domain + "/cdn-cgi/access/certs"`, cache the client, and call:

```python
claims = jwt.decode(
    token,
    signing_key.key,
    algorithms=["RS256"],
    audience=config.policy_audience,
    issuer=config.team_domain,
    options={"require": ["aud", "exp", "iat", "nbf", "iss", "type"]},
)
```

Convert all PyJWT errors to `AccessDenied("invalid Access JWT")` without
including the token or claims in the message.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```bash
python3 -m unittest tests.test_tts_access_auth -v
```

Expected: all configuration, employee-domain, service-allowlist, signature,
issuer, audience, and expiry tests pass.

- [ ] **Step 6: Commit**

```bash
git add tts_access_gateway tests/test_tts_access_auth.py
git commit -m "feat(auth): validate Cloudflare Access identities"
```

---

### Task 2: TTS-only request policy

**Files:**
- Create: `tts_access_gateway/policy.py`
- Test: `tests/test_tts_access_policy.py`

- [ ] **Step 1: Write failing route and payload tests**

```python
def test_only_tts_routes_are_allowed(self):
    validate_route("POST", "/v1/audio/speech")
    validate_route("GET", "/v1/models")
    with self.assertRaises(PolicyDenied):
        validate_route("POST", "/v1/chat/completions")

def test_payload_is_tts_model_voice_length_and_mp3_only(self):
    validate_speech_payload({
        "model": MODEL,
        "input": "hello",
        "voice": "Vivian",
        "response_format": "mp3",
    })
    for mutation in (
        {"model": "qwen3.6-27b"},
        {"voice": "clone-me"},
        {"input": "x" * 3001},
        {"response_format": "wav"},
    ):
        payload = {
            "model": MODEL,
            "input": "hello",
            "voice": "Vivian",
            "response_format": "mp3",
            **mutation,
        }
        with self.assertRaises(PolicyDenied):
            validate_speech_payload(payload)
```

- [ ] **Step 2: Run and verify RED**

Run:

```bash
python3 -m unittest tests.test_tts_access_policy -v
```

Expected: import failure because `policy.py` does not exist.

- [ ] **Step 3: Implement the exact policy**

```python
MODEL = "qwen3-tts-1.7b-customvoice"
VOICES = frozenset({
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric",
    "Ryan", "Aiden", "Ono_Anna", "Sohee",
})

def validate_route(method: str, path: str) -> None:
    if (method, path) not in {
        ("POST", "/v1/audio/speech"),
        ("GET", "/v1/models"),
    }:
        raise PolicyDenied("route not allowed")

def validate_speech_payload(payload: Mapping[str, object]) -> None:
    if payload.get("model") != MODEL:
        raise PolicyDenied("model not allowed")
    if payload.get("voice") not in VOICES:
        raise PolicyDenied("voice not allowed")
    text = payload.get("input")
    if not isinstance(text, str) or not text.strip() or len(text) > 3000:
        raise PolicyDenied("input must contain 1-3000 characters")
    if payload.get("response_format", "mp3") != "mp3":
        raise PolicyDenied("only MP3 is allowed")
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
python3 -m unittest tests.test_tts_access_policy -v
```

Expected: all tests pass.

```bash
git add tts_access_gateway/policy.py tests/test_tts_access_policy.py
git commit -m "feat(auth): enforce TTS-only request policy"
```

---

### Task 3: Authenticated streaming proxy and audit logging

**Files:**
- Create: `tts_access_gateway/app.py`
- Create: `tts_access_gateway/__main__.py`
- Create: `requirements-tts-access.txt`
- Test: `tests/test_tts_access_app.py`

- [ ] **Step 1: Write failing aiohttp integration tests**

Create local fake JWKS/auth and LiteLLM upstream fixtures. Assert:

```python
async def test_proxy_strips_client_auth_and_streams_mp3(self):
    response = await self.client.post(
        "/v1/audio/speech",
        headers={
            "Cf-Access-Jwt-Assertion": "signed-access-jwt",
            "Authorization": "Bearer employee-oauth-token",
            "X-Forwarded-For": "spoofed",
        },
        json={
            "model": MODEL,
            "input": "hello",
            "voice": "Vivian",
            "response_format": "mp3",
        },
    )
    self.assertEqual(200, response.status)
    self.assertEqual("audio/mpeg", response.headers["Content-Type"])
    self.assertEqual(b"ID3-test", await response.read())
    self.assertEqual(
        "Bearer internal-tts-key",
        self.upstream_request.headers["Authorization"],
    )
    self.assertNotIn("Cf-Access-Jwt-Assertion", self.upstream_request.headers)
    self.assertNotEqual(
        "spoofed",
        self.upstream_request.headers.get("X-Forwarded-For"),
    )
```

Also assert missing JWT is 403, invalid JSON is 400, body over 64 KiB is 413,
chat route is 403, `GET /v1/models` returns only the TTS model, and logs omit
text, instructions, token, and audio.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest tests.test_tts_access_app -v
```

Expected: import failure because `app.py` does not exist.

- [ ] **Step 3: Implement the aiohttp app**

```python
async def speech(request: web.Request) -> web.StreamResponse:
    request_id = request.headers.get("X-Request-Id") or str(uuid.uuid4())
    principal = await verifier.verify(
        request.headers.get("Cf-Access-Jwt-Assertion", "")
    )
    payload = await request.json()
    validate_speech_payload(payload)
    headers = {
        "Authorization": f"Bearer {config.litellm_api_key}",
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
    }
    async with upstream.post(
        f"{config.litellm_base_url}/v1/audio/speech",
        json=payload,
        headers=headers,
        timeout=timeout,
    ) as source:
        response = web.StreamResponse(
            status=source.status,
            headers={
                "Content-Type": source.headers.get(
                    "Content-Type", "application/json"
                ),
                "X-Request-Id": request_id,
            },
        )
        await response.prepare(request)
        async for chunk in source.content.iter_chunked(64 * 1024):
            await response.write(chunk)
        await response.write_eof()
        return response
```

Use `client_max_size=64 * 1024`, a 900-second total upstream timeout, a
bounded connector, and structured JSON audit records containing only actor,
request ID, route, model, voice, status, latency, and output bytes.

- [ ] **Step 4: Run focused and full tests**

```bash
python3 -m unittest tests.test_tts_access_app -v
python3 -m unittest discover -s tests -v
```

Expected: proxy tests and all existing gateway tests pass.

- [ ] **Step 5: Commit**

```bash
git add tts_access_gateway requirements-tts-access.txt tests/test_tts_access_app.py
git commit -m "feat(auth): proxy authenticated TTS requests"
```

---

### Task 4: Process supervision and Open WebUI trusted-header login

**Files:**
- Create: `run_tts_access_gateway.sh`
- Create: `systemd/llm-tts-access-gateway.service`
- Modify: `systemd/llm-gateway.target`
- Modify: `systemd/install.sh`
- Modify: `run_open_webui.sh`
- Modify: `tests/test_model_manager_tts.py`

- [ ] **Step 1: Add failing launcher and SSO configuration tests**

```python
def test_tts_auth_gateway_is_part_of_target(self):
    target = (ROOT / "systemd/llm-gateway.target").read_text()
    self.assertIn("llm-tts-access-gateway.service", target)

def test_open_webui_delegates_auth_to_cloudflare(self):
    launcher = (ROOT / "run_open_webui.sh").read_text()
    self.assertIn(
        'export WEBUI_AUTH_TRUSTED_EMAIL_HEADER="Cf-Access-Authenticated-User-Email"',
        launcher,
    )
    self.assertIn("export ENABLE_SIGNUP=False", launcher)
    self.assertIn("export ENABLE_LOGIN_FORM=False", launcher)
    self.assertIn("export ENABLE_PASSWORD_AUTH=False", launcher)
```

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest tests.test_model_manager_tts -v
```

Expected: new trusted-header and systemd assertions fail.

- [ ] **Step 3: Implement process and web settings**

`run_tts_access_gateway.sh` must require
`TTS_ACCESS_TEAM_DOMAIN`, `TTS_ACCESS_POLICY_AUD`, and
`TTS_GATEWAY_LITELLM_KEY`, activate `.venv-tts-access`, and execute:

```bash
exec "$SCRIPT_DIR/.venv-tts-access/bin/python" -m tts_access_gateway
```

The unit binds only through the application default
`127.0.0.1:8910`, uses `gateway.env`, restarts on failure, and appends to
`logs/tts-access-gateway.log`.

In `run_open_webui.sh`, set:

```bash
export WEBUI_AUTH_TRUSTED_EMAIL_HEADER="Cf-Access-Authenticated-User-Email"
export ENABLE_SIGNUP=False
export ENABLE_LOGIN_FORM=False
export ENABLE_PASSWORD_AUTH=False
```

Keep `WEBUI_AUTH=True`, the loopback bind, existing user data, and the
server-side LiteLLM key.

- [ ] **Step 4: Run tests and syntax checks**

```bash
python3 -m unittest discover -s tests -v
bash -n run_tts_access_gateway.sh run_open_webui.sh systemd/install.sh
```

Expected: all tests and shell syntax checks pass.

- [ ] **Step 5: Commit**

```bash
git add run_tts_access_gateway.sh run_open_webui.sh systemd tests/test_model_manager_tts.py
git commit -m "feat(auth): supervise TTS gateway and delegate web login"
```

---

### Task 5: Idempotent Cloudflare Access provisioning

**Files:**
- Create: `scripts/provision_cloudflare_access.py`
- Create: `tests/test_cloudflare_access_provision.py`

- [ ] **Step 1: Write failing payload and reconciliation tests**

Assert exact payloads for:

```python
organization_payload() == {
    "auth_domain": "stardust.cloudflareaccess.com",
    "name": "Stardust",
    "auto_redirect_to_identity": True,
    "session_duration": "24h",
}
```

```python
employee_policy_payload() == {
    "name": "Stardust employees",
    "decision": "allow",
    "include": [{"email_domain": {"domain": "stardust.ai"}}],
}
```

The TTS application must enable Managed OAuth, dynamic localhost and loopback
registration, a 15-minute access token, and a seven-day session. Reconciliation
must update matching resources by name/domain, never create duplicates, and
must print only resource IDs, names, domains, audiences, and status.

- [ ] **Step 2: Run and verify RED**

```bash
python3 -m unittest tests.test_cloudflare_access_provision -v
```

Expected: import failure because the provisioning module does not exist.

- [ ] **Step 3: Implement a minimal Cloudflare REST client**

Use `urllib.request`, read only `CLOUDFLARE_API_TOKEN` and
`CLOUDFLARE_ACCOUNT_ID`, and call:

- `/access/organizations`
- `/access/identity_providers`
- `/access/apps`
- `/access/apps/{app_id}/policies`

Create the OTP provider as:

```python
{"name": "One-time PIN login", "type": "onetimepin", "config": {}}
```

Create separate apps for `llm.preseen.ai` and `tts-api.preseen.ai`.
The TTS app uses:

```python
"oauth_configuration": {
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
```

Support `--check` and `--apply`; default to `--check`. After apply, GET
every resource and fail unless all expected values read back.

- [ ] **Step 4: Run tests and commit**

```bash
python3 -m unittest tests.test_cloudflare_access_provision -v
git add scripts/provision_cloudflare_access.py tests/test_cloudflare_access_provision.py
git commit -m "feat(auth): provision Cloudflare Access policy"
```

---

### Task 6: Managed OAuth and secure token storage in stardust-tts

**Files:**
- Create: `/Users/derek/Documents/Projects/stardust-skills/skills/stardust-tts/scripts/access_oauth.py`
- Create: `/Users/derek/Documents/Projects/stardust-skills/skills/stardust-tts/tests/test_access_oauth.py`
- Modify: `/Users/derek/Documents/Projects/stardust-skills/skills/stardust-tts/scripts/synthesize.py`
- Modify: `/Users/derek/Documents/Projects/stardust-skills/skills/stardust-tts/tests/test_synthesize.py`

- [ ] **Step 1: Write failing OAuth discovery, PKCE, refresh, and Keychain tests**

Tests must assert:

- RFC 9728 resource metadata is read from `WWW-Authenticate`.
- PKCE verifier has 43-128 URL-safe characters and challenge is S256.
- Dynamic registration uses a loopback callback and no client secret.
- OAuth state mismatch rejects the callback.
- Refresh is attempted before interactive login.
- macOS Keychain commands receive token data through stdin, not argv.
- non-macOS secure-store absence never writes a token file.
- service mode sends the two Cloudflare service headers.

Run:

```bash
python3 -m unittest discover -s skills/stardust-tts/tests -v
```

Expected: import failure for `access_oauth.py`.

- [ ] **Step 2: Implement the OAuth client**

Implement focused functions:

```python
discover_resource(url: str) -> ResourceMetadata
register_client(metadata: ResourceMetadata, redirect_uri: str) -> ClientRegistration
create_pkce() -> PkcePair
authorize_interactively(...) -> TokenSet
refresh_tokens(...) -> TokenSet
load_keychain(service: str, account: str) -> TokenCache | None
save_keychain(service: str, account: str, cache: TokenCache) -> None
delete_keychain(service: str, account: str) -> None
```

Use `webbrowser.open`, a random loopback port, constant-time state comparison,
and a five-minute callback timeout. Mask all token-bearing errors.

- [ ] **Step 3: Change synthesis authentication**

Set:

```python
DEFAULT_BASE_URL = "https://tts-api.preseen.ai/v1"
```

Authentication order:

1. If both `CF_ACCESS_CLIENT_ID` and `CF_ACCESS_CLIENT_SECRET` exist, send
   service headers.
2. Otherwise load/refresh or interactively acquire a Managed OAuth bearer token.
3. Never read `STARDUST_TTS_API_KEY` or `LITELLM_API_KEY`.

Add `--auth-status` and `--logout`. Never print token values.

- [ ] **Step 4: Run tests and commit in stardust-skills**

```bash
python3 -m unittest discover -s skills/stardust-tts/tests -v
python3 /Users/derek/.agents/skills/skill-creator/scripts/quick_validate.py skills/stardust-tts
git diff --check
git add skills/stardust-tts
git commit -m "feat(stardust-tts): authenticate with Cloudflare Access"
```

Expected: all tests and structural validation pass.

---

### Task 7: Documentation, complete regression, and code publication

**Files:**
- Modify: `README.md`
- Modify: `/Users/derek/Documents/Projects/stardust-skills/README.md`
- Modify: `/Users/derek/Documents/Projects/stardust-skills/skills/stardust-tts/SKILL.md`
- Modify: `/Users/derek/Documents/Projects/stardust-skills/skills/stardust-tts/README.md`

- [ ] **Step 1: Document exact employee and service flows**

Document:

```bash
python3 ~/.agents/skills/stardust-tts/scripts/synthesize.py \
  "欢迎使用星尘语音服务。" \
  --voice Vivian \
  --instructions "温暖、自然、语速稍慢" \
  --output /absolute/path/welcome.mp3
```

Document `--auth-status`, `--logout`, service-token environment names,
audit fields, token lifetimes, no-secret boundary, deployment, monitoring,
rollback, and the removal of `STARDUST_TTS_API_KEY`.

- [ ] **Step 2: Run all repository checks**

```bash
python3 -m unittest discover -s tests -v
bash -n run_*.sh systemd/install.sh
git diff --check
```

In stardust-skills:

```bash
python3 -m unittest discover -s skills/stardust-tts/tests -v
bash tests/install.test.sh
bash tests/sync-from-agents.test.sh
bash tests/sync-to-agents.test.sh
git diff --check
```

- [ ] **Step 3: Commit documentation and push both repositories**

```bash
git add README.md docs systemd scripts tts_access_gateway tests
git commit -m "docs: operate company-authenticated TTS"
git push origin master
```

```bash
git add README.md skills/stardust-tts tests/install.test.sh
git commit -m "docs(stardust-tts): explain employee OAuth access"
git push origin main
```

Read back both remote SHAs and require equality with local HEAD.

---

### Task 8: Production deployment and acceptance

**Files:**
- Modify on GPU4: `/home/stardust/preseen-ai-gateway/ingress.json`
- Deploy from llm-gateway: systemd unit, venv, and environment values

- [ ] **Step 1: Snapshot live state without secrets**

Capture Git SHAs, dirty status, systemd status, current ingress JSON, LiteLLM and
Open WebUI versions, public HTTP status, GPU status, and existing TTS/chat
smoke results. Do not print environment values.

- [ ] **Step 2: Create the isolated runtime and install**

```bash
python3 -m venv .venv-tts-access
.venv-tts-access/bin/pip install -r requirements-tts-access.txt
systemd/install.sh
```

Generate a new TTS-only LiteLLM virtual key through the authenticated local
management endpoint, store it only in `gateway.env` as
`TTS_GATEWAY_LITELLM_KEY`, and verify its model allowlist by readback.

- [ ] **Step 3: Provision Access**

Run `scripts/provision_cloudflare_access.py --check`, then `--apply` using
the existing secret API-token file without printing the token. If the token
lacks either required permission, stop with the exact missing Cloudflare
permission:

- `Access: Organizations, Identity Providers, and Groups Write`
- `Access: Apps and Policies Write`

Read back organization, OTP provider, apps, policies, Managed OAuth settings,
and application audience tags.

- [ ] **Step 4: Deploy protected ingress**

Edit only the relevant objects in the dirty local ingress source. Read the TTS
application AUD from the provisioning readback and use `apply_patch` to add a
`tts-api.preseen.ai -> http://127.0.0.1:8910` ingress whose
`originRequest.access` has `required: true`, `teamName: "stardust"`, and
`audTag` containing that complete read-back value. Add the analogous
read-back AUD to `llm.preseen.ai`. Do not derive, abbreviate, or copy an AUD
from logs. Sync ingress, create DNS, and read back tunnel configuration and DNS
without overwriting unrelated dirty changes.

- [ ] **Step 5: Start gateway and protect Open WebUI**

Start `llm-tts-access-gateway.service`, verify loopback health, then restart
Open WebUI after Access is live. Confirm no service binds a new LAN/public port.

- [ ] **Step 6: Run production acceptance**

Require:

- Unauthenticated TTS receives 401 with `WWW-Authenticate` and
  `resource_metadata`.
- Real employee OTP/PKCE creates a decodable MP3.
- Token refresh succeeds without another OTP during the grant session.
- Approved service identity succeeds; an unapproved one fails.
- Spoofed origin header, wrong AUD, expired JWT, chat route, wrong model, WAV,
  invalid voice, and 3001-character input fail.
- Open WebUI trusted-header login succeeds without local password and preserves
  existing roles/data.
- TTS cold start, GPU allocation/release, and a post-TTS chat request pass.
- Audit logs contain actor metadata and none of the prohibited content.

- [ ] **Step 7: Remove old public static-key route**

After all known callers use the new endpoint, remove
`llm-api.preseen.ai` from public tunnel ingress, sync, and require public
requests to fail independently of any old LiteLLM key. Loopback LiteLLM and
Open WebUI remain healthy.

- [ ] **Step 8: Notify the learning group**

Send a DingTalk message to the exact internal group `学习群` with the new
Skill URL, OTP login flow, example, MP3/voice capabilities, and a warning not to
share service credentials. Query the returned `openTaskId` until
`sendStatus=SUCCESS`.

- [ ] **Step 9: Final readback**

Report exact commands before results, Git SHAs, service status, Access resources,
public endpoint behavior, MP3 evidence, Open WebUI SSO, GPU regression, old
endpoint closure, notification delivery, known limitations, and rollback
commit/config snapshot.
