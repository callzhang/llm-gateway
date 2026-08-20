#!/usr/bin/env bash
# Run Open WebUI — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
OWUI_BIN_DIR="/home/derek/miniforge3/envs/llm-gateway-owui/bin"

# ── OpenAI-compatible backend: LiteLLM proxy on :8900 ─────────────────────────
export OPENAI_API_BASE_URL="http://127.0.0.1:8900/v1"
# Use the scoped "open-webui" virtual key, NOT the admin master key.  This keeps
# the frontend from being able to reach LiteLLM's /key/* /user/* admin routes.
# No fallback default — this file is tracked in a public repo.
: "${OPENWEBUI_LLM_KEY:?set OPENWEBUI_LLM_KEY in gateway.env (scoped virtual key)}"
export OPENAI_API_KEY="$OPENWEBUI_LLM_KEY"

# ── Text-to-speech: Qwen3-TTS through the same authenticated LiteLLM ──────────
# Open WebUI's audio subsystem has a separate OpenAI-compatible connection.
# Reuse its scoped virtual key; never give the UI LiteLLM's master key.
export AUDIO_TTS_ENGINE=openai
export AUDIO_TTS_OPENAI_API_BASE_URL="http://127.0.0.1:8900/v1"
export AUDIO_TTS_OPENAI_API_KEY="$OPENWEBUI_LLM_KEY"
export AUDIO_TTS_MODEL=qwen3-tts-1.7b-customvoice
export AUDIO_TTS_VOICE=Vivian

# ── Disable Ollama probing (no Ollama running) ────────────────────────────────
export ENABLE_OLLAMA_API=False

# ── Data / secret ─────────────────────────────────────────────────────────────
export DATA_DIR="$SCRIPT_DIR/open-webui-data"
# WEBUI_SECRET_KEY injected from gateway.env via systemd EnvironmentFile

# ── Isolate from LiteLLM's PostgreSQL DATABASE_URL (gateway.env) ─────────────
# Open WebUI uses SQLite by default; unset to prevent it picking up Postgres URL
unset DATABASE_URL

# ── Auth: switch to Cloudflare identity only after Access readback succeeds ─
export WEBUI_AUTH=True
export DEFAULT_USER_ROLE=user
if [ "${CLOUDFLARE_ACCESS_AUTH_ENABLED:-False}" = "True" ]; then
    # The origin is loopback-only. Access authenticates @stardust.ai and injects
    # this header; Open WebUI creates/maps the corresponding local account.
    export WEBUI_AUTH_TRUSTED_EMAIL_HEADER=Cf-Access-Authenticated-User-Email
    export WEBUI_AUTH_TRUSTED_NAME_HEADER=Cf-Access-Authenticated-User-Email
    export ENABLE_SIGNUP=False
    export ENABLE_LOGIN_FORM=False
    export ENABLE_PASSWORD_AUTH=False
else
    # Safe migration default: retain the existing domain-restricted login until
    # Access apps and policies have been created and read back successfully.
    export SIGNUP_ALLOWED_EMAIL_DOMAINS="stardust.ai"
    if "$OWUI_BIN_DIR/python" "$SCRIPT_DIR/scripts/ensure_owui_signup_patch.py"; then
        export ENABLE_SIGNUP=True
    else
        echo "[run_open_webui] signup domain allowlist could not be verified; disabling signup" >&2
        export ENABLE_SIGNUP=False
    fi
    export ENABLE_LOGIN_FORM=True
    export ENABLE_PASSWORD_AUTH=True
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
exec "$OWUI_BIN_DIR/open-webui" serve --host 127.0.0.1 --port 8080
