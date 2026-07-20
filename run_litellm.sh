#!/usr/bin/env bash
# Run LiteLLM proxy — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"   # make prisma CLI visible for schema migration
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-local-gateway}"
# No OPENAI_API_KEY: every model in config.yaml is served locally via
# model_manager on 127.0.0.1:8002.  There is deliberately no external fallback.

# Bind to loopback only — direct LAN access to the model API is closed.
# Reached internally by Open WebUI (and any local client) via 127.0.0.1:8900.
exec "$SCRIPT_DIR/.venv/bin/litellm" \
  --config "$SCRIPT_DIR/config.yaml" \
  --port 8900 \
  --host 0.0.0.0
