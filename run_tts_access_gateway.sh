#!/usr/bin/env bash
# Cloudflare Access identity bridge for the public TTS-only API.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TTS_ACCESS_TEAM_DOMAIN:?set TTS_ACCESS_TEAM_DOMAIN in gateway.env}"
: "${TTS_ACCESS_POLICY_AUD:?set TTS_ACCESS_POLICY_AUD in gateway.env}"
: "${TTS_GATEWAY_LITELLM_KEY:?set TTS_GATEWAY_LITELLM_KEY in gateway.env}"

export TTS_GATEWAY_HOST=127.0.0.1
export TTS_GATEWAY_PORT=8910
export TTS_GATEWAY_UPSTREAM_BASE_URL=http://127.0.0.1:8900

exec "$SCRIPT_DIR/.venv-tts-access/bin/python" -m tts_access_gateway
