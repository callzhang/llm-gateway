#!/usr/bin/env bash
# Run model_manager lazy-load proxy — foreground for systemd
exec /home/derek/Projects/gemma4-bench/.venv/bin/python \
  /home/derek/Projects/llm-gateway/model_manager.py
