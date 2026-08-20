# Employee TTS through a Skill and @stardust.ai Access login

Status: **live**. Provisioned, deployed, and verified end to end by a real
`@stardust.ai` login (see [Acceptance](#acceptance)).

> **Amended 2026-08-20.** This document originally argued for replacing Managed
> OAuth with a `cloudflared` dependency. That decision was reversed the next day
> — see [Why the cloudflared detour was reversed](#why-the-cloudflared-detour-was-reversed).
> The employee client is once again a self-contained OAuth client and the
> `2026-08-18` design's Managed OAuth choice stands after all. Everything below
> about the origin, the gateway, and the Access application is unchanged and
> still current.

## Goal

An ordinary Stardust employee installs one Claude Code skill, authenticates once
in a browser with their `@stardust.ai` email, and can synthesize speech. No
LiteLLM key is distributed, no key is stored on the employee's machine, and no
GPU4 credential leaves GPU4.

## Why the cloudflared detour was reversed

On 2026-08-19 this work replaced the 302-line Managed OAuth client
(`access_oauth.py`) with a 140-line wrapper around `cloudflared access login`.
The stated reasons were that the CLI is Cloudflare-maintained and that a
loopback + PKCE flow is security-sensitive code better not owned in-house.

Both reasons were wrong, and the cost was not accounted for.

- **A CLI can run the browser login itself.** That is RFC 8252, the standard
  native-app flow, and it is what `gh`, `aws`, `gcloud`, and `cloudflared`
  itself all implement. PKCE is a SHA-256 of a random string; the callback is
  `http.server` on a random loopback port; state validation is a string
  comparison. Calling this "security-sensitive code we cannot maintain" dressed
  a standard pattern up as hazardous engineering.
- **"Cloudflare maintains it better" had no supporting evidence here.** The four
  failures this deployment actually hit were a 1010 User-Agent block, a team
  rename invalidating live sessions, a fake-IP proxy on the client, and a stale
  org cookie. All environmental. `cloudflared` prevented none of them and was
  party to three.
- **The comparison used a self-imposed constraint as if it were a fact.** The
  argument against the OAuth client leaned on "session persistence is macOS
  only". That was true of that file — it called `_require_keychain()` and
  refused to run elsewhere — but it is not a property of the approach.
  `cloudflared` itself persists tokens as plain `0600` files in
  `~/.cloudflared/`; the same trust model was always available.
- **What remained was a real install step** for every employee, on every
  platform, forever.

The client is therefore self-contained again, with the one genuine defect of
the original fixed: the refresh token now lives in a `0600` file under the
user's config directory (`STARDUST_TTS_TOKEN_FILE` overrides it), written via
temp-file-and-rename so concurrent invocations cannot observe a partial
credential. Employees install nothing but the skill.

Consequences that remain true from the original design:

- The TTS Access application needs Managed OAuth with **loopback** client
  registration. localhost registration stays off, and the registration endpoint
  refuses non-loopback redirect URIs — both verified.
- Unauthenticated API requests now get an RFC 9728 discovery `401` with a
  `WWW-Authenticate: Bearer ... resource_metadata=...` header rather than a 302
  to the login page. That is the correct behaviour for API clients and is what
  the 2026-08-18 design specified.
- Because the redirect lands on `127.0.0.1`, the browser must run on the same
  machine as the skill. Over SSH the sign-in URL is printed instead of opened.

## Scope

In scope: `tts-api.preseen.ai`, the `tts-auth-gateway` on `127.0.0.1:8910`, the
`stardust-tts` skill, and the Access application and policy that protect that
one hostname.

Explicitly out of scope: putting `llm.preseen.ai` (Open WebUI) behind Access.
That is a separate cutover with its own blast radius — it must move in lockstep
with `CLOUDFLARE_ACCESS_AUTH_ENABLED` in `run_open_webui.sh` or every existing
chat user is locked out. `scripts/provision_cloudflare_access.py` therefore
grows an `--only` selector instead of always reconciling both applications, and
this work runs it as `--only tts`.

Also out of scope: removing the `llm-api.preseen.ai` static-key route. Its
public virtual key `public-internet-llm-api-v2` is scoped to the three chat
models and cannot reach `qwen3-tts-1.7b-customvoice`, so it is not a TTS bypass.

## Employee flow

```text
Employee runs the skill
  -> stored refresh token, or a browser login the skill opens itself
       (RFC 8252: dynamic registration -> PKCE -> loopback redirect -> email OTP)
  -> POST https://tts-api.preseen.ai/v1/audio/speech
       header: Authorization: Bearer <OAuth access token>
  -> Cloudflare Access policy: email domain == stardust.ai
  -> edge injects Cf-Access-Jwt-Assertion
  -> tts-auth-gateway 127.0.0.1:8910   (verifies RS256, iss, aud, exp/iat/nbf,
                                        type == app, sub, email domain)
  -> internal TTS-only LiteLLM virtual key
  -> LiteLLM 127.0.0.1:8900 /v1/audio/speech
  -> model_manager 127.0.0.1:8002 -> vLLM-Omni Qwen3-TTS on a GPU slot
  -> audio/mpeg
```

The employee never sees a LiteLLM key. The refresh token is the only stored
credential, in a `0600` file under the user's config directory.

### Claim shape

`principal_from_claims` requires `type == "app"`, a non-empty `sub`, and an
`email` whose domain is exactly `stardust.ai`. Confirmed against a real token on
2026-08-19: `type: "app"`, `sub: ad7c6439-…`, `email: dev-accounts@stardust.ai`,
`iss: https://stardust-ai.cloudflareaccess.com`, `aud` a single-element list
holding the application audience, 24h lifetime. No change to
`principal_from_claims` was needed.

For the same reason the skill presents the token both as a `cf-access-token`
header and as a `CF_Authorization` cookie. The edge ignores whichever it is not
using, and guessing wrong would be expensive to diagnose: every call would come
back as the HTML login page, which is indistinguishable from an expired session
and sends the employee into `--login` attempts that cannot help.

## Gateway changes

These are defects found while reviewing the committed implementation against the
design document, fixed here.

1. **Upstream failure breaks the JSON contract.** `error_middleware` did not
   catch `aiohttp.ClientError` or `asyncio.TimeoutError`, so a LiteLLM restart —
   routine on this host — produced aiohttp's default HTML 500 on a public
   endpoint and left no audit record. Upstream connection and timeout failures
   now return a JSON 502/504 in the documented error shape. This satisfies
   "LiteLLM or GPU4 failures are returned without switching provider".

2. **Denials were not audited.** `_audit` only ran on the success path, so a
   403 or 400 against the public hostname left no trace at all — `access_log` is
   `None`, making the audit logger the only record. The design's acceptance
   criterion "audit logs identify the actor" is unmet for exactly the requests
   that matter most. Every terminal response is now audited. A denied request
   has no verified principal, so it is recorded as `actor: null`,
   `actor_kind: "unauthenticated"`, plus the denial reason. Reasons are drawn
   from our own policy and auth messages, never from request content, so the
   privacy rule (no input text, instructions, tokens, or audio bytes in logs)
   still holds.

3. **Dead environment variables.** `run_tts_access_gateway.sh` exported
   `TTS_GATEWAY_HOST`, `TTS_GATEWAY_PORT`, and `TTS_GATEWAY_UPSTREAM_BASE_URL`;
   `config.py` read none of them (listen host and port were dataclass defaults
   and the upstream key was `TTS_GATEWAY_LITELLM_BASE_URL`). It was correct only
   because the defaults happened to match, and would have silently ignored a
   port change during deployment. `config.py` now reads the names the launcher
   exports, and keeps `TTS_GATEWAY_LITELLM_BASE_URL` as a deprecated alias.

4. **Unbounded restart churn.** The unit cannot start before the Access
   application exists, because `TTS_ACCESS_POLICY_AUD` is only known after
   provisioning. That ordering is inherent. Restarting every 10 seconds forever
   is not: the unit accumulated 1096 failed starts. `StartLimitIntervalSec` /
   `StartLimitBurst` now park the unit after five failures so a missing-config
   state is visible as `failed` instead of hiding in a restart loop.

Header hygiene (design step 7) was already satisfied and is left alone:
`_handle` builds `upstream_headers` as a fresh dict and passes it as `headers=`,
so no client `Authorization`, Cloudflare identity, forwarding, or hop-by-hop
header is copied toward LiteLLM.

## Cold start and the Cloudflare response cap

`IDLE_TIMEOUT` is 300s and TTS shares it with the chat models, so five idle
minutes unload the model and the next request pays a measured **56s** cold
start. Measured warm latency is 0.17–0.20s for a short phrase and **20.2s** for
2910 characters.

`/v1/audio/speech` is not streamed — the whole MP3 arrives in one response.

**Measured 2026-08-20, correcting an earlier assumption in this document:** a
cold call took **104.2s** at the gateway and returned 200 through Cloudflare.
An earlier revision claimed Cloudflare's non-streaming proxied response budget
was roughly 100s and that a cold call risked a 524. That figure was carried over
untested from an unrelated note; it has now been exercised and contradicted.
Do not design around a ~100s edge cap on this path without re-measuring.

What remains true is the user-facing cost: the first call after five idle
minutes takes roughly a minute and a half with no progress output. Warm calls
are ~1.0s.

Decision: **do not** pin the TTS model warm. It costs ~30 GiB of VRAM on a card
the 27B and 35B models need, to serve a bursty, low-volume interactive endpoint.
The skill sets a 900s client timeout, so a cold start cannot time out client
side, and it warns the user before a cold call.

If interactive use grows enough that 56s waits become a complaint, the cheaper
lever is a TTS-specific idle timeout in `model_manager`, not a permanent
keep-warm.

## The skill

The employee client is **`stardust-tts` in
<https://github.com/stardustai/stardust-skills>**, not in this repository. That
skill already existed and already targeted `tts-api.preseen.ai`; what changed
here is only its authentication layer.

It shipped a Managed OAuth client (`scripts/access_oauth.py`: RFC 9728
discovery, dynamic registration, PKCE, macOS Keychain). That path cannot
authenticate against the application as provisioned, because Managed OAuth is
off. It was replaced by `scripts/access_login.py`, which shells out to
`cloudflared access login` / `access token`. `scripts/synthesize.py` keeps its
CLI, its local validation (nine voices, 3000 characters, MP3-only), its
`audio/mpeg` and MP3-signature checks, and its atomic write; only the three
auth call sites moved.

Two consequences worth noting:

- Interactive use is no longer macOS-only. The old client refused to run
  interactively without Keychain; `cloudflared` carries the credential on every
  platform it supports.
- The headless path is unchanged. `CF_ACCESS_CLIENT_ID` /
  `CF_ACCESS_CLIENT_SECRET` still take precedence, and those client IDs are what
  `TTS_ACCESS_SERVICE_CLIENT_IDS` allowlists at the gateway.

Validating locally as well as at the gateway is deliberate: it turns a
round-trip and a 403 into an immediate, specific message. There is no key-based
path for employees, by design.

## Runbook

Everything below needs the Cloudflare API token in
`/home/stardust/preseen-ai-gateway/cf-api.env`, which this session could not
read. These steps are for an operator to run.

```bash
export CLOUDFLARE_API_TOKEN=$(sudo grep '^CLOUDFLARE_API_TOKEN=' \
  /home/stardust/preseen-ai-gateway/cf-api.env | cut -d= -f2-)
export CLOUDFLARE_ACCOUNT_ID=32040609845cecfd444552c934feb7fb
```

1. **Inspect, then provision the TTS application only.** *(Done 2026-08-19 through the Zero Trust dashboard, because the API token has Access read but not edit — `auth.forbidden (code 1010)` on every write.)*

   ```bash
   cd ~/Projects/llm-gateway
   .venv/bin/python scripts/provision_cloudflare_access.py --only tts --check
   .venv/bin/python scripts/provision_cloudflare_access.py --only tts --apply
   ```

   `--check` mutates nothing. `--apply` reconciles the Access organization,
   the one-time-PIN identity provider, the `Stardust TTS API` application on
   `tts-api.preseen.ai`, and the `Stardust employees`
   (`email_domain: stardust.ai`) policy. It does **not** touch
   `llm.preseen.ai`. Record the printed `aud`.

   The team domain is **`stardust-ai.cloudflareaccess.com`**. Zero Trust
   auto-generated `late-scene-8b71` on enablement; it was renamed the same day,
   which is free while no device is enrolled and no external IdP is integrated,
   and expensive afterwards. It is not
   `stardust.cloudflareaccess.com`: that name, and `preseen.cloudflareaccess.com`,
   both resolve but belong to other organizations. `cloudflareaccess.com` is a
   global namespace, so pointing `TTS_ACCESS_TEAM_DOMAIN` at the wrong one would
   make the gateway fetch a foreign JWKS and accept tokens that organization
   signed, including ones minted for our `aud`. Read it, never guess it.

2. **Create the DNS record.** *(Done 2026-08-19.)* Tunnel ingress already
   mapped `tts-api.preseen.ai -> 127.0.0.1:8910`, so only the proxied CNAME was
   missing:

   ```bash
   cloudflared tunnel login   # once, for the zone
   cloudflared tunnel route dns c76aa6f8-de21-4df8-8cae-0636f9805d34 tts-api.preseen.ai
   ```

3. **Configure and start the gateway.** `TTS_ACCESS_TEAM_DOMAIN` is already in
   `gateway.env`; only the audience from step 1 is missing, and the unit is
   stopped until it is there.

   ```bash
   echo 'TTS_ACCESS_POLICY_AUD=<aud from step 1>' >> ~/Projects/llm-gateway/gateway.env
   systemctl --user start llm-tts-access-gateway.service
   systemctl --user status llm-tts-access-gateway.service
   ```

4. **Verify the claim shape before announcing anything.**

   ```bash
   cloudflared access login --app https://tts-api.preseen.ai
   cloudflared access token --app https://tts-api.preseen.ai \
     | cut -d. -f2 | base64 -d 2>/dev/null | python3 -m json.tool
   ```

   Confirm `type == "app"`, a non-empty `sub`, and `email` ending in
   `@stardust.ai`. If any differs, fix `principal_from_claims` in
   `tts_access_gateway/auth.py` and re-run the test suite before continuing.

5. **End-to-end as an employee.**

   ```bash
   python3 ~/.agents/skills/stardust-tts/scripts/synthesize.py \
     "验证通过" --voice Vivian --output /tmp/v.mp3
   file /tmp/v.mp3     # expect: MPEG ADTS, layer III
   ```

6. **Negative checks.** No token → Access login page, no origin response. A
   non-`stardust.ai` identity → 403. `qwen3.6-27b` as the model, or
   `/v1/chat/completions` as the route → 403 before LiteLLM. Confirm each
   denial produced an audit line with `actor_kind: "unauthenticated"` or the
   policy reason in `logs/tts_access_gateway.log`.

7. **Publish.** Tell the DingTalk learning group how to install the skill and
   that first use opens a browser for an `@stardust.ai` one-time PIN.

## Acceptance

All verified on 2026-08-19, on GPU4 and through the public hostname.

Backend:

- Speech works through LiteLLM: 200, `audio/mpeg`, decodable MP3
  (`MPEG ADTS, layer III, v2, 64 kbps, 24 kHz, monaural`).
- Warm latency 0.17–0.20s short, 20.2s at 2910 characters; cold start 56s.
- Open WebUI's read-aloud request shape (`speed`, `response_format: "wav"`) is
  normalized to MP3 and returns 200.

Edge and identity:

- An unauthenticated request to `tts-api.preseen.ai` gets 302 to
  `stardust-ai.cloudflareaccess.com`; the origin never sees it.
- A real `@stardust.ai` one-time-PIN login through
  `cloudflared access login` yields an application JWT whose claims match
  `principal_from_claims` exactly.
- The skill synthesized through the full chain in 1.9s warm, 43632 bytes,
  `MPEG ADTS, layer III` — no LiteLLM key anywhere on the client.
- The success is audited as
  `actor: dev-accounts@stardust.ai`, `actor_kind: "employee"`, with model,
  voice, status, latency, and byte count — and no input text.

The self-contained OAuth client was verified on a real employee machine
(macOS, 2026-08-21): the skill opened the browser itself, the `@stardust.ai`
one-time code completed the flow, the refresh token landed in
`~/.config/stardust-tts/oauth.json` with mode `0600`, and the output was a
decodable `MPEG ADTS, layer III` file. Discovery, loopback client registration,
and refusal of non-loopback redirect URIs were verified separately against the
live application.

Browser Integrity Check is disabled for `tts-api.preseen.ai` only, matched on
hostname. A valid User-Agent alone is not enough: BIC also weighs client IP
reputation, so an identical request passed from a US host and was blocked with
`403 error code: 1010` from a CN one.

Fails closed, each rejected before LiteLLM and each audited:

| Request | Result |
|---|---|
| `model: qwen3.6-27b` | 403 `model not allowed` |
| `POST /v1/chat/completions` | 403 `route not allowed` |
| `voice: Morgan` | 403 `voice not allowed` |
| no `Cf-Access-Jwt-Assertion` | 403 `access denied`, audited as unauthenticated |

Route denials are audited as `unauthenticated` because `validate_route` runs
before verification; payload denials carry the employee identity because
authentication has already succeeded by then. Both are correct and intentional.

Tests: 74 pass (was 38).

## One edge behaviour worth remembering

The first real skill call failed with `403 error code: 1010` — Cloudflare's
browser-integrity check rejecting urllib's default `Python-urllib/3.12`
User-Agent *before Access was consulted*. curl with the same token worked. The
skill now sends an identifiable User-Agent, and reports 1010 as a client-signature
problem rather than telling the employee to sign in again — which would have sent
them round a loop that cannot help.

## Rollback

Stop `llm-tts-access-gateway.service`, remove the `tts-api.preseen.ai` DNS
record, and delete the `Stardust TTS API` Access application. Nothing else is
touched: LiteLLM, model_manager, the TTS backend, Open WebUI, its login mode,
tunnel ingress for every other hostname, and all existing keys are unchanged,
because this work deliberately never enabled Access on `llm.preseen.ai`.
