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

# ── RAG: use the upstream LLM for embeddings; skip local model download ───────
export RAG_EMBEDDING_ENGINE=openai
export RAG_EMBEDDING_MODEL=qwen3.6-27b

# ── Listen ────────────────────────────────────────────────────────────────────
export HOST=0.0.0.0
export PORT=3000

exec "$SCRIPT_DIR/.venv-owui/bin/open-webui" serve
