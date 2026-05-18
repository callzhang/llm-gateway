#!/usr/bin/env bash
# Run LLM Gateway — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LITELLM_MASTER_KEY="${LITELLM_MASTER_KEY:-sk-local-gateway}"
export LITELLM_BASE="${LITELLM_BASE:-http://127.0.0.1:8901}"

exec "$SCRIPT_DIR/.venv/bin/python" "$SCRIPT_DIR/gateway.py"
