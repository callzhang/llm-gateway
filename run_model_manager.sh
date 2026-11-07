#!/usr/bin/env bash
# Run model_manager lazy-load proxy — foreground for systemd
export PYTHONNOUSERSITE=1
PYTHON_BIN=${PYTHON_BIN:-/home/derek/miniforge3/envs/llm-gateway-vllm/bin/python}
exec "$PYTHON_BIN" "$(dirname "${BASH_SOURCE[0]}")/model_manager.py"
