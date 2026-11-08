#!/usr/bin/env bash
# Direct TLS entrance of the CN region: llm-cn.preseen.ai -> the A100 site's
# public IP, port-forwarded by the site's NAT to the A100's Caddy
# (systemd/region-cn/Caddyfile).  No Cloudflare proxy: the A100's overseas
# traffic leaves through a VPN in Tokyo and its tunnel lands in Los Angeles, so
# a CN client going through Cloudflare crosses the Pacific twice.
#
#   sudo -E scripts/region_cn_tls.sh dns    # once: DNS-only A record -> CN_PUBLIC_IP
#   sudo scripts/region_cn_tls.sh renew     # issue, or renew within 30 days of expiry; push if changed
#   scripts/region_cn_tls.sh push           # copy the cert in CERT_DIR to the A100, restart Caddy
#
# `renew` runs daily from llm-cn-tls-renew.timer (systemd/system).
#
# The certificate is issued here on GPU4, never on the A100: issuing needs the
# Cloudflare API token (DNS:Edit on the whole zone), and six other users have
# sudo on the A100 host.
set -euo pipefail

ZONE_ID=c9f464a8e14df477cd7c39617f7b963f
HOST=${CN_TLS_HOST:-llm-cn.preseen.ai}
CN_PUBLIC_IP=${CN_PUBLIC_IP:-27.221.10.54}
CF_ENV=${CF_ENV:-/home/stardust/preseen-ai-gateway/cf-api.env}
CERT_DIR=${CERT_DIR:-/etc/llm-cn-tls/current}   # fullchain.pem + privkey.pem
A100=${A100:-a100-1}
A100_TLS_DIR=/data/derek/llm-gateway/tls
LEGO_DIR=${LEGO_DIR:-/etc/llm-cn-tls/lego}
PUSH_USER=${PUSH_USER:-derek}   # owns the ssh key that reaches the A100

case "${1:-}" in
  renew)
    [ "$(id -u)" = 0 ] || { echo "renew runs as root (reads $CF_ENV, writes /etc)" >&2; exit 1; }
    source "$CF_ENV"
    : "${CLOUDFLARE_API_TOKEN:?missing in $CF_ENV}"
    umask 077
    mkdir -p "$LEGO_DIR" "$CERT_DIR"
    before=$(sha256sum "$CERT_DIR/fullchain.pem" 2>/dev/null || true)
    # `run` both issues and renews; it is a no-op until 30 days before expiry.
    CLOUDFLARE_DNS_API_TOKEN=$CLOUDFLARE_API_TOKEN lego run --path "$LEGO_DIR" \
      --accept-tos --domains "$HOST" --key-type EC256 --dns cloudflare \
      --renew-days 30 --no-random-sleep
    src=$(find "$LEGO_DIR" -name "$HOST.crt" -print -quit)
    key=${src%.crt}.key
    [ -s "$src" ] && [ -s "$key" ] || { echo "lego produced no $HOST.crt/.key under $LEGO_DIR" >&2; exit 1; }
    install -m 600 "$src" "$CERT_DIR/fullchain.pem"
    install -m 600 "$key" "$CERT_DIR/privkey.pem"
    if [ "$before" != "$(sha256sum "$CERT_DIR/fullchain.pem")" ] || [ "${FORCE_PUSH:-}" = 1 ]; then
      runuser -u "$PUSH_USER" -- "$0" push
    else
      echo "certificate unchanged; nothing to push"
    fi
    ;;
  dns)
    source "$CF_ENV"
    : "${CLOUDFLARE_API_TOKEN:?missing in $CF_ENV}"
    api() { curl -fsS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" "$@"; }
    base="https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records"
    # proxied=false: the point is to bypass Cloudflare.
    body="{\"type\":\"A\",\"name\":\"$HOST\",\"content\":\"$CN_PUBLIC_IP\",\"proxied\":false,\"ttl\":300,\"comment\":\"CN region direct entrance (llm-gateway docs/region-cn.md)\"}"
    id=$(api "$base?type=A&name=$HOST" | python3 -c 'import sys,json;r=json.load(sys.stdin)["result"];print(r[0]["id"] if r else "")')
    if [ -n "$id" ]; then api -X PUT "$base/$id" --data "$body" >/dev/null
    else api -X POST "$base" --data "$body" >/dev/null; fi
    echo "$HOST A $CN_PUBLIC_IP (DNS only)"
    ;;
  push)
    # Run as the user whose ssh key reaches the A100; reads CERT_DIR via sudo.
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    sudo cat "$CERT_DIR/fullchain.pem" > "$tmp/fullchain.pem"
    sudo cat "$CERT_DIR/privkey.pem" > "$tmp/privkey.pem"
    ssh "$A100" "umask 077; mkdir -p $A100_TLS_DIR && setfacl -b $A100_TLS_DIR 2>/dev/null; chmod 700 $A100_TLS_DIR"
    scp -q "$tmp/fullchain.pem" "$tmp/privkey.pem" "$A100:$A100_TLS_DIR/"
    ssh "$A100" "setfacl -b $A100_TLS_DIR/*.pem 2>/dev/null; chmod 600 $A100_TLS_DIR/*.pem; XDG_RUNTIME_DIR=/run/user/\$(id -u) systemctl --user restart llm-tls-proxy"
    echo "pushed $(sudo openssl x509 -noout -enddate -in "$CERT_DIR/fullchain.pem") to $A100"
    ;;
  *)
    sed -n '2,14p' "$0"; exit 2 ;;
esac
