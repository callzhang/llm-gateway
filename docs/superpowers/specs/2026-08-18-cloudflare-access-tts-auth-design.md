# Cloudflare Access authentication for Stardust TTS

## Goal

Replace employee-distributed LiteLLM API keys with company-email authentication
for Stardust TTS while preserving the existing GPU4 model lifecycle, Open WebUI,
and server-to-server operation.

The employee browser and public Skill must authenticate through Cloudflare
Access using an `@stardust.ai` email. Interactive API clients use Managed OAuth
Authorization Code with PKCE. Headless workloads use individually revocable
Cloudflare service identities. LiteLLM credentials remain only on GPU4.

## Current state

- `llm.preseen.ai` reaches Open WebUI on `127.0.0.1:8080` through the
  host-wide Cloudflare Tunnel.
- Open WebUI currently uses local email/password accounts. A local patch permits
  self-registration only for `stardust.ai`, but this is not SSO.
- `llm-api.preseen.ai` reaches LiteLLM on `127.0.0.1:8900` and accepts
  LiteLLM bearer keys from the public Internet.
- LiteLLM 1.86.2 has no JWT/OIDC authentication configured.
- Cloudflare Zero Trust Access is not yet enabled for the account.
- The Cloudflare tunnel source is
  `/home/stardust/preseen-ai-gateway`; it has unrelated uncommitted changes
  that must be preserved.

## Chosen architecture

The Cloudflare Access team domain is
`https://stardust.cloudflareaccess.com`. Access uses the built-in email
one-time PIN identity provider and permits only `@stardust.ai` identities.

### Browser path

```text
Employee browser
  -> Cloudflare Access email OTP
  -> llm.preseen.ai
  -> Cf-Access-Authenticated-User-Email
  -> Open WebUI trusted-header authentication
  -> LiteLLM loopback with the existing server-side scoped key
```

Open WebUI trusts only `Cf-Access-Authenticated-User-Email`. It continues to
bind to loopback, and cloudflared also validates the Access application audience
before proxying. Local signup, the local login form, and password authentication
are disabled after trusted-header acceptance passes. Existing users retain
their roles because Open WebUI matches accounts by normalized email.

### Employee Skill path

```text
Employee Skill
  -> Cloudflare Managed OAuth discovery
  -> dynamic public-client registration
  -> browser OTP + Authorization Code with PKCE
  -> short-lived OAuth access token
  -> tts-api.preseen.ai
  -> Cloudflare Access policy
  -> signed Cf-Access-Jwt-Assertion
  -> tts-auth-gateway on 127.0.0.1:8910
  -> internal TTS-only LiteLLM key
  -> LiteLLM /v1/audio/speech
  -> GPU4 Qwen3-TTS
```

Managed OAuth access tokens last 15 minutes. The grant session and refresh token
last seven days. Policy is re-evaluated on refresh. On macOS, the Skill stores
the refresh token and dynamic client registration in Keychain. It never writes
tokens to the Skill directory, environment variables, command arguments, or a
plain-text file. Platforms without an implemented secure store use in-memory
tokens only and require a new login after expiry.

### Headless path

Each workload receives its own Cloudflare Access Service Token. It authenticates
with `CF-Access-Client-Id` and `CF-Access-Client-Secret`; Cloudflare attaches
an application JWT to the origin request. Service identities are individually
named, expired, rotated, audited, and revoked. They never reuse an employee
token, Open WebUI key, LiteLLM master key, or another workload's token.

## TTS authentication gateway

The new `tts-auth-gateway` is an aiohttp service on
`127.0.0.1:8910`. It is the only origin for `tts-api.preseen.ai`.

For every request it:

1. Requires `Cf-Access-Jwt-Assertion`.
2. Verifies RS256 signature against
   `https://stardust.cloudflareaccess.com/cdn-cgi/access/certs`.
3. Verifies issuer, application audience, `exp`, `iat`, and `nbf`.
4. Accepts either:
   - an employee JWT with non-empty `sub` and normalized email whose domain is
     exactly `stardust.ai`; or
   - a service JWT whose `common_name` is in the configured allowlist.
5. Allows only `POST /v1/audio/speech` and restricted
   `GET /v1/models`.
6. Allows only `qwen3-tts-1.7b-customvoice`, the nine verified preset voices,
   at most 3000 Unicode characters, and MP3 output.
7. Removes client authorization, Cloudflare identity, forwarding, and
   hop-by-hop headers.
8. Adds the internal TTS-only LiteLLM virtual key and proxies to
   `http://127.0.0.1:8900`.
9. Streams a successful `audio/mpeg` response without buffering it into logs.

The gateway has explicit body-size, request timeout, concurrency, and response
size safeguards. It does not expose chat, embeddings, management routes, other
models, WAV output, voice cloning, VoiceDesign, or provider fallback.

JWKS is cached with bounded TTL and refreshed when an unknown `kid` appears.
A network failure may use an unexpired cached key. If no valid key can verify the
request, authentication fails closed.

## Audit and privacy

Audit records contain request ID, normalized employee email or service client
ID, timestamp, path, model, voice, HTTP status, latency, and output byte count.
They never contain input text, instructions, OAuth tokens, service secrets,
LiteLLM keys, Cloudflare JWTs, or audio bytes.

The employee email is retained because it is the operational identity required
for access review and revocation. Logs use the existing service log retention
and file permissions.

## Repository changes

### llm-gateway

- Add the authentication gateway in a focused module.
- Add unit and integration tests with local fake JWKS and LiteLLM upstreams.
- Add a pinned runtime dependency file and runner.
- Add a user systemd unit and include it in `llm-gateway.target`.
- Add trusted-header Open WebUI environment settings and fail-closed startup
  assertions.
- Document provisioning, operations, audit fields, rollback, and acceptance.

### stardust-skills

- Change `stardust-tts` to use `https://tts-api.preseen.ai/v1`.
- Remove employee dependence on `STARDUST_TTS_API_KEY`.
- Add Managed OAuth discovery, dynamic registration, PKCE, local callback,
  token refresh, Keychain storage, logout, and auth status.
- Add explicit headless service-token mode.
- Keep local text, voice, length, MP3, and response validation.
- Add unit tests that never contact Cloudflare or store real credentials.

### preseen-ai-gateway

- Enable the `stardust.cloudflareaccess.com` Access organization.
- Configure the one-time PIN identity provider.
- Add self-hosted applications and policies for `llm.preseen.ai` and
  `tts-api.preseen.ai`.
- Enable Managed OAuth, dynamic loopback registration, 15-minute access tokens,
  and seven-day grant sessions for the TTS application.
- Add `tts-api.preseen.ai -> 127.0.0.1:8910` to tunnel ingress.
- Require the correct Access audience in cloudflared origin settings.
- Preserve all existing unrelated changes in the dirty local repository.

## Failure behavior

- Missing employee OAuth returns Cloudflare's RFC 9728 discovery 401.
- Invalid or expired OAuth is rejected at the edge.
- Missing, invalid, expired, wrong-issuer, or wrong-audience application JWT is
  rejected by the origin with 403.
- Wrong email domain or unapproved service identity is rejected with 403.
- Unsupported route or model is rejected before LiteLLM.
- Invalid TTS input is rejected locally by the Skill and again by the gateway.
- LiteLLM or GPU4 failures are returned without switching provider.
- An unavailable JWKS endpoint with no valid cache rejects all requests.
- Missing auth configuration prevents the gateway from starting.

## Deployment sequence

1. Commit this design and the implementation plan.
2. Develop the gateway and Skill with test-first changes.
3. Deploy and test the gateway only on loopback.
4. Enable Access and provision both applications and policies.
5. Add `tts-api.preseen.ai` and verify OAuth plus service identity.
6. Publish the updated Skill.
7. Enable trusted-header login for Open WebUI and verify existing roles.
8. Run production acceptance for TTS, chat regression, cold start, and GPU
   release.
9. Remove the public static-key `llm-api.preseen.ai` ingress after all known
   callers have migrated.
10. Notify the DingTalk learning group with usage and security guidance.

No step may claim company-only access until the old public static-key hostname
has been removed or independently restricted by Access.

## Acceptance criteria

- An unauthenticated TTS API request receives OAuth discovery and no origin
  response.
- A real `@stardust.ai` OTP login produces a valid, decodable MP3.
- Refresh succeeds after access-token expiry without exposing the refresh token.
- Invalid domain, audience, signature, expiry, model, voice, route, and spoofed
  identity header tests all fail closed.
- An approved service identity can synthesize; an unapproved identity cannot.
- Open WebUI performs company OTP login without a second local password.
- Existing Open WebUI user roles and data remain intact.
- Audit logs identify the actor and contain none of the prohibited payloads.
- The public static-key hostname is no longer usable after cutover.
- TTS cold start, MP3 output, chat model wake-up, and GPU release regressions
  pass.

## Rollback

Rollback restores the previous cloudflared ingress snapshot and Open WebUI local
login environment, disables the new Access applications, and stops
`tts-auth-gateway`. LiteLLM, model_manager, the TTS backend, existing data, and
the internal Open WebUI key are unchanged. The previous public static-key route
may be restored only as a time-bounded emergency measure with the scoped key
rotated and an explicit follow-up to remove it.
