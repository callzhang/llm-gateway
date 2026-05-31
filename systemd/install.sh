#!/usr/bin/env bash
# Install and enable LLM Gateway systemd user services (no sudo needed after install).
# Run once with sudo to enable-linger; all subsequent management is via systemctl --user.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SYSTEMD="$HOME/.config/systemd/user"
SERVICES=(
  llm-model-manager.service
  llm-litellm.service
  llm-open-webui.service
)

# Remove old system-level units if present
LEGACY=(
  llm-gemma4.service
  llm-qwen36-27b.service
  llm-qwen36-35b.service
  llm-ocr-middleware.service
  llm-model-manager.service
  llm-litellm.service
)
for svc in "${LEGACY[@]}"; do
  if [ -f "/etc/systemd/system/$svc" ]; then
    sudo systemctl disable --now "$svc" 2>/dev/null || true
    sudo rm -f "/etc/systemd/system/$svc"
    echo "  removed system-level $svc"
  fi
done
sudo systemctl daemon-reload 2>/dev/null || true

# Enable linger so user services survive logout
sudo loginctl enable-linger "$USER"
echo "  linger enabled for $USER"

mkdir -p "$USER_SYSTEMD"
mkdir -p "$HOME/Projects/llm-gateway/logs"

echo "Installing user service files..."
for svc in "${SERVICES[@]}"; do
  cp "$SCRIPT_DIR/$svc" "$USER_SYSTEMD/"
  echo "  copied $svc → $USER_SYSTEMD/"
done

systemctl --user daemon-reload

echo "Enabling services..."
for svc in "${SERVICES[@]}"; do
  systemctl --user enable "$svc"
  echo "  enabled $svc"
done

echo ""
echo "Done. Manage with:"
echo "  systemctl --user start   llm-model-manager llm-litellm llm-open-webui"
echo "  systemctl --user status  llm-model-manager llm-litellm llm-open-webui"
echo "  systemctl --user restart llm-litellm llm-open-webui"
echo "  systemctl --user stop    llm-open-webui llm-litellm llm-model-manager"
echo ""
echo "  Chat UI: http://$(hostname -I | awk '{print $1}'):3000"
