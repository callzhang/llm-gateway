#!/usr/bin/env bash
# Run LiteLLM proxy — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$SCRIPT_DIR/.venv/bin:$PATH"   # make prisma CLI visible for schema migration
# No fallback default: the key must come from gateway.env (gitignored).  A
# hardcoded default here is a tracked file in a PUBLIC repo, so it would be a
# published credential the moment gateway.env is missing.
: "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY in gateway.env (openssl rand -hex 32)}"
export LITELLM_MASTER_KEY
# No OPENAI_API_KEY: every model in config.yaml is served locally via
# model_manager on 127.0.0.1:8002.  There is deliberately no external fallback.

# Binds 0.0.0.0 so the Cloudflare tunnel and loopback clients can both reach it.
# Direct LAN access is closed by ufw (default DROP, no rule for 8900), NOT by
# the bind address.  Public traffic arrives only via llm-api.preseen.ai through
# the cloudflared tunnel, authenticated by a LiteLLM virtual key.
exec "$SCRIPT_DIR/.venv/bin/litellm" \
  --config "$SCRIPT_DIR/config.yaml" \
  --port 8900 \
  --host 0.0.0.0
