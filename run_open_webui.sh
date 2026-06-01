#!/usr/bin/env bash
# Run Open WebUI — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── OpenAI-compatible backend: LiteLLM proxy on :8900 ─────────────────────────
export OPENAI_API_BASE_URL="http://127.0.0.1:8900/v1"
export OPENAI_API_KEY="${LITELLM_MASTER_KEY:-sk-local-gateway}"

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
# Keep open signup enabled (the company-domain allowlist above is the gate).
export ENABLE_SIGNUP=True

# ── RAG: use the upstream LLM for embeddings; skip local model download ───────
export RAG_EMBEDDING_ENGINE=openai
export RAG_EMBEDDING_MODEL=qwen3.6-27b

# ── Listen ────────────────────────────────────────────────────────────────────
# NOTE: `open-webui serve` ignores the PORT/HOST env vars — it only honours the
# --host/--port flags (default 8080). :3000 is permanently taken by the Langfuse
# Docker container, so Open WebUI runs on :8080.
# Bind to loopback only — the sole external entrypoint is the Cloudflare tunnel
# (llm.preseen.ai), which runs on this host and reaches us via 127.0.0.1.
exec "$SCRIPT_DIR/.venv-owui/bin/open-webui" serve --host 127.0.0.1 --port 8080
