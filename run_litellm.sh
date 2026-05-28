#!/usr/bin/env bash
# Run LiteLLM proxy — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"   # make prisma CLI visible for schema migration
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-local-gateway}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

exec "$SCRIPT_DIR/.venv/bin/litellm" \
  --config "$SCRIPT_DIR/config.yaml" \
  --port 8900 \
  --host 0.0.0.0
