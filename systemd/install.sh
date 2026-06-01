#!/usr/bin/env bash
# Install and enable LLM Gateway systemd user services (no sudo needed after install).
# Run once with sudo to enable-linger; all subsequent management is via systemctl --user.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
USER_SYSTEMD="$HOME/.config/systemd/user"
TARGET=llm-gateway.target
SERVICES=(
  llm-model-manager.service
  llm-litellm.service
  llm-open-webui.service
)
# NOTE: the public Cloudflare tunnel was moved out of this stack into a single
# host-wide root system service (see ~/Projects/preseen-ai-gateway). It is no
# longer a member of llm-gateway.target.
UNITS=("${SERVICES[@]}" "$TARGET")

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

echo "Installing user unit files (services + target)..."
for unit in "${UNITS[@]}"; do
  cp "$SCRIPT_DIR/$unit" "$USER_SYSTEMD/"
  echo "  copied $unit → $USER_SYSTEMD/"
done

systemctl --user daemon-reload

# Clear any stale enablement from a previous layout (services used to be
# WantedBy=default.target; they are now WantedBy=llm-gateway.target).
echo "Refreshing enablement..."
systemctl --user disable "${SERVICES[@]}" >/dev/null 2>&1 || true

# Enable the services (→ symlinked under llm-gateway.target.wants) and the
# target itself (→ symlinked under default.target.wants, starts on boot).
systemctl --user enable "${SERVICES[@]}"
systemctl --user enable "$TARGET"
echo "  enabled ${SERVICES[*]} and $TARGET"

echo ""
echo "Done. The whole stack is now one unit — manage with the target:"
echo "  systemctl --user start    $TARGET     # bring up the whole stack"
echo "  systemctl --user stop     $TARGET     # tear down the whole stack"
echo "  systemctl --user restart  $TARGET     # restart everything"
echo "  systemctl --user status   $TARGET     # group status"
echo ""
echo "Individual components are still independently controllable, e.g.:"
echo "  systemctl --user restart  llm-litellm"
echo "  journalctl --user -u      llm-open-webui -f"
echo ""
echo "  Chat UI (public): https://llm.preseen.ai"
echo "  (public tunnel is the host-wide root service 'cloudflared' —"
echo "   source: ~/Projects/preseen-ai-gateway · sudo systemctl status cloudflared)"
