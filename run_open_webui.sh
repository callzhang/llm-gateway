#!/usr/bin/env bash
# Run Open WebUI — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── OpenAI-compatible backend: LiteLLM proxy on :8900 ─────────────────────────
export OPENAI_API_BASE_URL="http://127.0.0.1:8900/v1"
# Use the scoped "open-webui" virtual key, NOT the admin master key.  This keeps
# the frontend from being able to reach LiteLLM's /key/* /user/* admin routes.
# No fallback default — this file is tracked in a public repo.
: "${OPENWEBUI_LLM_KEY:?set OPENWEBUI_LLM_KEY in gateway.env (scoped virtual key)}"
export OPENAI_API_KEY="$OPENWEBUI_LLM_KEY"

# ── Disable Ollama probing (no Ollama running) ────────────────────────────────
export ENABLE_OLLAMA_API=False

# ── Data / secret ─────────────────────────────────────────────────────────────
export DATA_DIR="$SCRIPT_DIR/open-webui-data"
# WEBUI_SECRET_KEY injected from gateway.env via systemd EnvironmentFile

# ── Isolate from LiteLLM's PostgreSQL DATABASE_URL (gateway.env) ─────────────
# Open WebUI uses SQLite by default; unset to prevent it picking up Postgres URL
unset DATABASE_URL

# ── Auth: require login; disable open signup after first admin registers ──────
export WEBUI_AUTH=True

# ── Company-domain allowlist for self-signup (enforced by LOCAL PATCH in ──────
#    open_webui/routers/auths.py). Only these email domains may register;
#    the very first user (initial admin) is exempt. Comma-separated.
export SIGNUP_ALLOWED_EMAIL_DOMAINS="stardust.ai"
# New self-signup users become active immediately (no admin approval step).
export DEFAULT_USER_ROLE=user
# That allowlist lives in .venv-owui/ — pip-managed and gitignored — so any
# `pip install -U open-webui` overwrites auths.py and silently reopens public
# signup on llm.preseen.ai.  Re-assert the patch on every start, and FAIL
# CLOSED: if it cannot be verified in place, serve with signup off rather than
# with an unguarded registration form.  Existing users keep working either way.
if "$SCRIPT_DIR/.venv-owui/bin/python" "$SCRIPT_DIR/scripts/ensure_owui_signup_patch.py"; then
    export ENABLE_SIGNUP=True
else
    echo "[run_open_webui] !! signup domain allowlist could NOT be verified" >&2
    echo "[run_open_webui] !! disabling signup (fail-closed); existing logins unaffected" >&2
    echo "[run_open_webui] !! re-derive the anchor in scripts/ensure_owui_signup_patch.py" >&2
    export ENABLE_SIGNUP=False
fi

# ── RAG embeddings: Jina embedding service (OpenAI-compatible) ────────────────
# Open WebUI's RAG embedding uses its OWN OpenAI endpoint (RAG_OPENAI_API_*),
# NOT the chat OPENAI_API_BASE_URL above.  Without RAG_OPENAI_API_BASE_URL it
# silently defaults to https://api.openai.com/v1 → 401.  Point it at the
# canonical jina-embeddings-v5-text-small endpoint (embed.preseen.ai) — the
# stable service URL, not a raw loopback port that can shuffle between providers.
# Note: qwen3.6-27b is a CHAT model and cannot embed — that was also wrong.
export RAG_EMBEDDING_ENGINE=openai
export RAG_EMBEDDING_MODEL=jinaai/jina-embeddings-v5-text-small
export RAG_OPENAI_API_BASE_URL="https://embed.preseen.ai/v1"
# Key injected from gateway.env via systemd EnvironmentFile (kept out of git).
export RAG_OPENAI_API_KEY="${RAG_OPENAI_API_KEY:-}"
# Service has no per-request batch cap (concurrency_limit 64); 32 verified OK.
export RAG_EMBEDDING_BATCH_SIZE=32

# ── Listen ────────────────────────────────────────────────────────────────────
# NOTE: `open-webui serve` ignores the PORT/HOST env vars — it only honours the
# --host/--port flags (default 8080). :3000 is permanently taken by the Langfuse
# Docker container, so Open WebUI runs on :8080.
# Bind to loopback only — the sole external entrypoint is the Cloudflare tunnel
# (llm.preseen.ai), which runs on this host and reaches us via 127.0.0.1.
exec "$SCRIPT_DIR/.venv-owui/bin/open-webui" serve --host 127.0.0.1 --port 8080
