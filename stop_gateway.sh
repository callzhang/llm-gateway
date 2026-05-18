#!/usr/bin/env bash
pkill -f "litellm.*config.yaml" && echo "LiteLLM stopped"      || echo "LiteLLM not running"
pkill -f "gateway\.py"          && echo "Gateway stopped"       || echo "Gateway not running"
pkill -f "model_manager.py"     && echo "Model manager stopped" || echo "Model manager not running"
pkill -f "vllm serve"           && echo "vLLM stopped"          || echo "vLLM not running"
