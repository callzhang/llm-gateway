#!/usr/bin/env bash
# Run LiteLLM proxy — foreground for systemd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
# GATEWAY_PY_BIN: the env holding litellm + prisma; another region sets its own.  The default is the
# venv of the version pinned in docs/litellm-*-requirements.txt, never the env shared with vLLM: that
# one still has LiteLLM 1.99.0, which with these configs logs other callers' requests.
GATEWAY_PY_BIN=${GATEWAY_PY_BIN:-/home/derek/venvs/litellm-1.108.19/bin}
export PATH="$GATEWAY_PY_BIN:$PATH"   # make prisma CLI visible for schema migration
# No fallback default: the key must come from gateway.env (gitignored).  A
# hardcoded default here is a tracked file in a PUBLIC repo, so it would be a
# published credential the moment gateway.env is missing.
: "${LITELLM_MASTER_KEY:?set LITELLM_MASTER_KEY in gateway.env (openssl rand -hex 32)}"
export LITELLM_MASTER_KEY
# What the config's deployments send to model_manager, here and in the peer region.
: "${GATEWAY_API_KEY:?set GATEWAY_API_KEY in gateway.env (the same value in both regions)}"
export GATEWAY_API_KEY
# No OPENAI_API_KEY: every model in config.yaml is served locally via
# model_manager on 127.0.0.1:8002.  There is deliberately no external fallback.

# GPU4 binds 0.0.0.0 so the Cloudflare tunnel and loopback clients can both
# reach it.  Direct LAN access is closed by ufw (default DROP, no rule for
# 8900), NOT by the bind address.  Public traffic arrives only via
# llm-api.preseen.ai through the cloudflared tunnel, authenticated by a LiteLLM
# virtual key.  The A100 host has no host firewall and other users on its LAN,
# so it sets LITELLM_HOST=127.0.0.1; its direct entrance is the TLS proxy
# (deploy/Caddyfile.cn), which carries the source-IP allowlist.
# Store prompts and responses in the spend logs whole. LiteLLM rebuilds a `previous_response_id`
# conversation from those rows (responses/.../session_handler.py), and by default it cuts every
# string past 2,048 characters and splices in "(litellm_truncated skipped N chars ...)", so a
# continued request reached the model with gutted instructions and earlier answers. On the A100
# (2026-11-17) CN memory's continued extractions thought 10-20k tokens and returned no answer in
# 87 of 142 calls. Retention (maximum_spend_logs_retention_period) still bounds the table.
export MAX_STRING_LENGTH_PROMPT_IN_DB="${MAX_STRING_LENGTH_PROMPT_IN_DB:-10000000}"
# LITELLM_CONFIG: each region has its own model list (config.cn.yaml on the A100).
exec "$GATEWAY_PY_BIN/litellm" \
  --config "$SCRIPT_DIR/${LITELLM_CONFIG:-config.yaml}" \
  --port 8900 \
  --host "${LITELLM_HOST:-0.0.0.0}"
