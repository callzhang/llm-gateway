#!/usr/bin/env bash
# Run model_manager lazy-load proxy — foreground for systemd
PYTHON_BIN=${PYTHON_BIN:-/home/derek/Projects/llm-gateway/.venv/bin/python}
exec "$PYTHON_BIN" /home/derek/Projects/llm-gateway/model_manager.py
