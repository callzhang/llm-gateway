# Reusing Cloudflare Access for other services

Short answer: **yes — the team domain is meant to be reused.** One Zero Trust
organization serves every service you own. What must *not* be reused is the
application, and that distinction is the whole of this document.

Written 2026-08-19, after putting `tts-api.preseen.ai` behind Access.

## What is shared and what is not

| Thing | Scope | Reused? |
|---|---|---|
| Team domain `stardust-ai.cloudflareaccess.com` | the whole account | **Yes** — one for everything |
| Identity provider (One-time PIN) | the whole account | **Yes** |
| JWKS at `<team domain>/cdn-cgi/access/certs` | the whole account | **Yes** — one key set signs every app's tokens |
| Employee seats (50 on the Free plan) | the whole account | **Yes** — shared pool |
| **Access application** | one hostname (or path) | **No — one per service** |
| **AUD tag** | one application | **No — never share** |
| Policy | attached per application | Reusable as a *reusable policy*, or copied |
| Session duration | per application | Per service |

The team domain is an identity endpoint, not a service endpoint. It is where
employees log in and where origins fetch signing keys. Ten services behind
Access means ten applications, one team domain, one login.

## Why the AUD must not be shared

An Access JWT carries `aud` naming the application it was issued for. An origin
that verifies tokens checks exactly two things about provenance: the issuer
(your team domain) and the audience (its own application).

If two services share one Access application, they share one AUD, and a token
minted for a visit to service A verifies perfectly at service B. Anyone allowed
into the least sensitive service is silently allowed into the most sensitive
one. Give every service its own application; that is what makes the audience
check mean anything.

The same logic explains why the *team domain* must be read, never guessed:
`cloudflareaccess.com` is a global namespace, and both
`stardust.cloudflareaccess.com` and `preseen.cloudflareaccess.com` are live and
belong to other organizations. An origin pointed at the wrong team domain
fetches a foreign JWKS and will accept tokens that organization signed —
including ones minted with your AUD.

## Two ways an origin can consume Access

### A. Verify the JWT yourself

What `tts_access_gateway` does. The edge injects `Cf-Access-Jwt-Assertion`; the
origin verifies RS256 against the team JWKS, checks `iss`, `aud`, `exp`/`iat`/
`nbf`, then maps claims to a principal.

Use this when the service is an API, needs per-request identity in its own
audit log, or must enforce policy the edge cannot express (route, model, payload
shape). It is the only option if the origin is reachable by anything other than
the tunnel.

The reusable part is small — roughly `auth.py` plus three environment
variables:

```
<SVC>_ACCESS_TEAM_DOMAIN=https://stardust-ai.cloudflareaccess.com   # shared
<SVC>_ACCESS_POLICY_AUD=<this service's AUD>                        # unique
<SVC>_ACCESS_SERVICE_CLIENT_IDS=a.access,b.access                   # optional
```

### B. Trust an edge-injected header

What Open WebUI would do with `Cf-Access-Authenticated-User-Email`. Far less
code, and correct **only** when the origin binds to loopback and the tunnel is
its sole path in. A header is trivially forged by anyone who can reach the
origin directly, so this pattern's safety rests entirely on that being
impossible.

Use it for browser-facing apps that already have their own session layer and
just need to know who the user is.

## Candidates on this host

Every hostname on the `preseen-gateway` tunnel is a candidate; none except
`tts-api` is behind Access today.

| Hostname | Origin | Today | Natural pattern |
|---|---|---|---|
| `tts-api.preseen.ai` | :8910 | **Access** | A (done) |
| `llm.preseen.ai` | :8080 Open WebUI | password + domain allowlist | B |
| `llm-api.preseen.ai` | :8900 LiteLLM | static virtual key | A, or leave for machines |
| `embed.preseen.ai` | :7997 | key | A |
| `ocr.preseen.ai` | :7998 | key | A |
| `gliner.preseen.ai` | :7999 | key | A |
| `video-transcribe.preseen.ai` | :8899 | key | A |
| `eval-tracking.preseen.ai` | :4319 | — | B |
| `mc-*-origin.preseen.ai` | :878x | service-to-service | service tokens, not employee login |

`llm.preseen.ai` is the one with real blast radius: cutting it over must happen
in the same breath as flipping `CLOUDFLARE_ACCESS_AUTH_ENABLED` in
`run_open_webui.sh`, or every existing chat user is locked out.

## Adding a service

1. **Create the application.** Zero Trust → 访问控制 → 应用程序 → 新建 →
   自托管. Set the hostname, 24h session, and an allow policy
   (`Emails ending in` → `@stardust.ai`). Leave Managed OAuth off unless a
   client genuinely needs dynamic registration — `cloudflared access login`
   does not.

   Or, once the API token carries `Access: Apps and Policies: Edit` and
   `Access: Organizations, Identity Providers, and Groups: Edit`:

   ```bash
   .venv/bin/python scripts/provision_cloudflare_access.py --only <name> --apply
   ```

   after adding the service to `APPLICATIONS` in that script.

2. **Record the AUD.** The dashboard shows it on the application's Overview.
   It is also the `kid` query parameter of the login redirect, so it can be read
   without dashboard access:

   ```bash
   curl -s -o /dev/null -w '%{redirect_url}\n' https://<host>/
   ```

3. **Point DNS at the tunnel**, if the hostname is new:

   ```bash
   cloudflared tunnel route dns c76aa6f8-de21-4df8-8cae-0636f9805d34 <host>
   ```

   and add the ingress rule in `/home/stardust/preseen-ai-gateway/ingress.json`,
   then `sudo ./sync-ingress.sh`.

4. **Wire the origin** using pattern A or B. For A, copy
   `tts_access_gateway/auth.py`; only the AUD differs.

5. **Verify before announcing.** An unauthenticated request must 302 to the team
   domain and the origin must never see it. Then log in and decode the token —
   `type` must be `app`, `sub` non-empty, `email` on the right domain, `iss` your
   team domain, `aud` this service's AUD.

## Clients

**Interactive.** Reuse `access_oauth.py` from the `stardust-tts` skill. It is a
self-contained RFC 8252 client — dynamic registration, PKCE, loopback redirect —
and it is already parameterised by service: every entry point takes `base_url`,
and the refresh token is stored per origin, so one file serves any number of
Access-protected hostnames. A second skill needs
`auth_headers("https://<host>/v1")` and nothing else.

Employees install nothing. An earlier revision of this document told you to
shell out to `cloudflared access login`; that was reversed on 2026-08-20
because it put a binary install in front of every employee and bought nothing
in return.

Because the redirect lands on `127.0.0.1`, the browser must run on the same
machine as the client. Over SSH the sign-in URL is printed rather than opened,
and the callback still has to reach that machine's loopback.

**Headless.** Create an Access *service token* per workload and send
`CF-Access-Client-Id` / `CF-Access-Client-Secret`. Each workload gets its own,
named and individually revocable. Add its `common_name` to the origin's
allowlist (`TTS_ACCESS_SERVICE_CLIENT_IDS` for TTS). Never let a workload reuse
an employee token.

**Send a real User-Agent — and know that it is not enough.** Cloudflare's
browser-integrity check answers `403 error code: 1010` to `Python-urllib/3.x`
*before Access is consulted*. But BIC also weighs client IP reputation: the
same request, with the same honest User-Agent, passed from a US host and was
blocked from a CN one. Fixing the User-Agent and declaring victory is exactly
the trap — it works on the machine you test from.

For an Access-protected API, turn BIC off for that hostname with a
Configuration Rule; Access already rejects anything without a valid JWT, so BIC
adds nothing and only breaks legitimate non-browser clients. **Match on the
hostname**, `http.host eq "<host>"`. A first attempt here matched
`full_uri contains "<host>"`, which let any URL in the zone disable BIC by
appending that string to its query string.

A 1010 looks exactly like an authentication failure and tempts you to re-login
forever; it is not one, and clients should say so rather than suggesting a
fresh login.

## Renaming the team domain

`late-scene-8b71` → `stardust-ai` on 2026-08-19. This is cheap **only** while no
device is enrolled in the Cloudflare One Client and no external IdP is
integrated; both must be reconfigured otherwise. Every origin's team-domain
setting must change in step. AUDs are unaffected — they belong to the
application, not the organization.

One thing to expect afterwards: `~/.cloudflared/cert.pem` was issued under the
old organization. Nothing here needed it again the same day, but
`cloudflared tunnel route dns` uses it, so expect to re-run
`cloudflared tunnel login` before adding the next hostname.

**A fake-IP proxy on the client breaks Access in a way that looks like a
server fault.** Clash/Surge-style tools answer DNS from the RFC 2544 range
`198.18.0.0/15` and dial the real host themselves. A client in that state gets
404 on `/cdn-cgi/access/*` and passes straight through to the origin, while the
same URL from an unproxied host is correctly intercepted — so the two vantage
points disagree and every server-side check looks healthy. The visible symptom
is Cloudflare's *"Unable to find your Access organization"* page, which reads
like a broken deployment.

Diagnose it in one command on the client, not by comparing edge behaviour:

```bash
dig +short tts-api.preseen.ai A     # 198.18.x.x means a local proxy, not DNS
```

Fix it by bypassing the proxy for the application hostname and for
`*.cloudflareaccess.com`. If a team shares a proxy configuration, expect every
one of them to hit this, and say so when announcing the service.

**The old team domain keeps resolving.** `late-scene-8b71.cloudflareaccess.com`
still answers 200 at the root after the rename — only `/cdn-cgi/access/certs`
404s — and its App Launcher renders *"Unable to find your Access organization.
Please enter a valid team name."* Any stale bookmark, open tab, or login link
pasted into chat before the rename now produces that page, which reads like a
broken deployment rather than a dead link. After renaming, go and invalidate the
old links people are holding.

Unrelated to renaming, but easy to misattribute to it: `cloudflared access
login` waits roughly seven minutes for the browser step and then fails with
`Could not create access request: failed to run transfer service: Failed to
fetch resource`. That message means nobody finished the login in time, not that
anything is misconfigured. Run the command when you are ready to click through
it, rather than starting it and coming back later.

Do it before onboarding anyone, or not at all.

## Sharing the client across skills

`install.sh` rsyncs each `skills/*` directory into `~/.agents/skills/`
independently, so there is no shared library today and a second skill cannot
simply import the first one's module. Three ways out, in order of preference:

1. **`skills/_shared/access_oauth.py`,** copied by `install.sh` alongside the
   skills, with each skill adding it to `sys.path`. One change to the installer,
   one copy of the auth code, no drift. This is the recommended shape.
2. **Vendor a copy per skill.** Works immediately, needs no installer change,
   and guarantees the copies diverge — the 1010 User-Agent fix had to be applied
   twice already, once in `synthesize.py` and once in `access_oauth.py`, inside
   a single skill.
3. **A pip-installable internal package.** Correct at scale, heaviest to set up,
   and adds an install step for employees — the thing this design just removed.

Whichever is chosen, the client itself needs no changes: it is already
service-agnostic.

## Limits worth knowing

- **50 seats** on Zero Trust Free. A seat is consumed by a user who
  authenticates, and released by the inactivity policy (currently 未启用 — turn
  it on before approaching the cap).
- **There is no ~100s cap** on non-streaming proxied responses, contrary to what
  this document and the TTS spec both previously asserted. That figure was
  carried untested from an unrelated note and it shaped design decisions. A cold
  TTS call measured 104.2s end to end and returned 200. Measure before designing
  around any edge timeout.
- **A session outlives a policy edit.** The token issued at login carries the
  matched `policy_id` and its own 24-hour expiry, and the origin verifies only
  signature, issuer, audience, and time. Treat removal from an allow policy as
  taking effect at the next login unless you revoke the session explicitly —
  confirm the exact revocation semantics in Cloudflare's docs before relying on
  either behaviour for an offboarding.
