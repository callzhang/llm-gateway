#!/usr/bin/env bash
# Install and enable the LLM Gateway systemd services (needs sudo).
#
# System units running as derek, not user units: systemd-oomd only spares root-owned cgroups from
# swap kills, and it killed the user-unit model_manager again and again (2026-11).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSTEM_SYSTEMD=/etc/systemd/system
TARGET=llm-gateway.target
SERVICES=(
  llm-jit-warmup.service
  llm-model-manager.service
  llm-litellm.service
  llm-open-webui.service
)
# NOTE: the public Cloudflare tunnel was moved out of this stack into a single
# host-wide root system service (see ~/Projects/preseen-ai-gateway). It is no
# longer a member of llm-gateway.target.
UNITS=("${SERVICES[@]}" "$TARGET")

# A user unit left beside its system twin would fight it for the GPUs and ports.
USER_SYSTEMD=/home/derek/.config/systemd/user
for unit in "${UNITS[@]}" llm-tts-access-gateway.service; do
  if [ -f "$USER_SYSTEMD/$unit" ] && [ ! -L "$USER_SYSTEMD/$unit" ]; then
    echo "ERROR: $USER_SYSTEMD/$unit is still installed. Stop it, move it away and"
    echo "       'systemctl --user mask $unit', then re-run."
    exit 1
  fi
done

mkdir -p "$SCRIPT_DIR/../logs"

# Log rotation.  Without it logs/ grows unbounded — gateway.log.1 once reached 5.1 GB.
LOGROTATE_SRC="$SCRIPT_DIR/../logrotate/llm-gateway"
if [ -f "$LOGROTATE_SRC" ]; then
  sudo cp "$LOGROTATE_SRC" /etc/logrotate.d/llm-gateway
  echo "  installed /etc/logrotate.d/llm-gateway"
fi

for unit in "${UNITS[@]}"; do
  sudo install -m 644 "$SCRIPT_DIR/$unit" "$SYSTEM_SYSTEMD/$unit"
  echo "  installed $unit"
done
# Drop-ins carry settings that must not be lost on reinstall — e.g. model_manager's
# JIT-warmup ordering, without which a cold FlashInfer cache gets compiled on the
# request path.
for dropin in "$SCRIPT_DIR"/*.service.d; do
  [ -d "$dropin" ] || continue
  name=$(basename "$dropin")
  sudo install -d "$SYSTEM_SYSTEMD/$name"
  sudo install -m 644 "$dropin"/*.conf "$SYSTEM_SYSTEMD/$name/"
  echo "  installed $name/*.conf"
done

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICES[@]}" "$TARGET"

echo ""
echo "Done. Manage the whole stack with the target:"
echo "  sudo systemctl start|stop|restart|status $TARGET"
echo "  sudo systemctl restart llm-litellm"
echo "  journalctl -u llm-open-webui -f"
echo ""
echo "  Chat UI (public): https://llm.preseen.ai"
