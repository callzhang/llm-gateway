#!/usr/bin/env bash
# Cloudflare tunnel `llm-api-geo`: llm-api.preseen.ai served by BOTH regions.
#
# One remotely-managed tunnel, one connector (replica) per region: GPU4 (US)
# and the A100 host (CN).  Cloudflare forwards each request to the replica
# closest to the Cloudflare data center that received it, which is the
# "nearest region" split; each region's LiteLLM then overflows to the other
# over Tailscale.  It is a separate tunnel from preseen-gateway because a
# replica serves every hostname of its tunnel — an A100 replica of
# preseen-gateway would send ocr/embed/... to the A100's own ports.
#
#   sudo -E scripts/llm_api_geo_tunnel.sh create         # once: tunnel + token files
#   sudo -E scripts/llm_api_geo_tunnel.sh sync           # push the ingress below
#   sudo -E scripts/llm_api_geo_tunnel.sh dns <hostname> # CNAME hostname -> this tunnel
#
# The API token (Tunnel:Edit + DNS:Edit) lives in preseen-ai-gateway/cf-api.env.
set -euo pipefail

ACCOUNT_ID=32040609845cecfd444552c934feb7fb
ZONE_ID=c9f464a8e14df477cd7c39617f7b963f
NAME=llm-api-geo
CF_ENV=${CF_ENV:-/home/stardust/preseen-ai-gateway/cf-api.env}
TOKEN_FILE=${TOKEN_FILE:-/etc/cloudflared/llm-api-geo.env}   # TUNNEL_TOKEN=..., root 600

source "$CF_ENV"
: "${CLOUDFLARE_API_TOKEN:?missing in $CF_ENV}"
api() { curl -fsS -H "Authorization: Bearer $CLOUDFLARE_API_TOKEN" -H "Content-Type: application/json" "$@"; }
base="https://api.cloudflare.com/client/v4/accounts/$ACCOUNT_ID/cfd_tunnel"

tunnel_id() {
  api "$base?name=$NAME&is_deleted=false" | python3 -c 'import sys,json;r=json.load(sys.stdin)["result"];print(r[0]["id"] if r else "")'
}

case "${1:-}" in
  create)
    id=$(tunnel_id)
    if [ -z "$id" ]; then
      id=$(api -X POST "$base" --data "{\"name\":\"$NAME\",\"config_src\":\"cloudflare\"}" \
           | python3 -c 'import sys,json;print(json.load(sys.stdin)["result"]["id"])')
    fi
    install -d -m 755 "$(dirname "$TOKEN_FILE")"
    umask 077
    api "$base/$id/token" | python3 -c 'import sys,json;print("TUNNEL_TOKEN="+json.load(sys.stdin)["result"])' > "$TOKEN_FILE"
    echo "tunnel $NAME = $id; token written to $TOKEN_FILE"
    ;;
  sync)
    id=$(tunnel_id)
    # Both replicas reach their own region's LiteLLM on loopback :8900.
    api -X PUT "$base/$id/configurations" --data '{"config":{"ingress":[
      {"hostname":"llm-api.preseen.ai","service":"http://127.0.0.1:8900"},
      {"hostname":"llm-api-geo.preseen.ai","service":"http://127.0.0.1:8900"},
      {"service":"http_status:404"}]}}' \
    | python3 -c 'import sys,json;d=json.load(sys.stdin);print("success",d["success"])'
    ;;
  dns)
    host=${2:?hostname}
    id=$(tunnel_id)
    target="$id.cfargotunnel.com"
    rec=$(api "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records?type=CNAME&name=$host" \
          | python3 -c 'import sys,json;r=json.load(sys.stdin)["result"];print(r[0]["id"] if r else "")')
    body="{\"type\":\"CNAME\",\"name\":\"$host\",\"content\":\"$target\",\"proxied\":true}"
    if [ -n "$rec" ]; then
      api -X PATCH "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records/$rec" --data "$body" >/dev/null
    else
      api -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE_ID/dns_records" --data "$body" >/dev/null
    fi
    echo "$host -> $target"
    ;;
  *) echo "usage: $0 create|sync|dns <hostname>" >&2; exit 2 ;;
esac
