#!/usr/bin/env bash
# Capture what LiteLLM is doing when its memory climbs, before the unit's
# MemoryHigh (3G) throttles it or MemoryMax (4G) kills it.
#
# Runs as root from llm-litellm-mem-watch.timer every minute: py-spy needs
# ptrace, and kernel.yama.ptrace_scope=1 forbids it for a non-child process
# unless the caller is root. Each capture goes to logs/mem-captures/<time>/.
set -euo pipefail

OWNER=derek
GATEWAY_DIR=/home/derek/Projects/llm-gateway
PY_SPY=/home/derek/miniforge3/envs/llm-gateway-vllm/bin/py-spy
THRESHOLD_BYTES=${THRESHOLD_BYTES:-2147483648}   # 2 GiB of memory.current
MIN_GAP_SECONDS=${MIN_GAP_SECONDS:-600}          # at most one capture per 10 min
CAPTURE_ROOT=$GATEWAY_DIR/logs/mem-captures

user_systemctl() { systemctl --user --machine="$OWNER@.host" "$@"; }

cgroup=$(user_systemctl show llm-litellm.service -p ControlGroup --value)
main_pid=$(user_systemctl show llm-litellm.service -p MainPID --value)
[ -n "$cgroup" ] && [ "$main_pid" != 0 ] || exit 0
cg_dir=/sys/fs/cgroup$cgroup
current=$(cat "$cg_dir/memory.current")
[ "$current" -ge "$THRESHOLD_BYTES" ] || exit 0

latest=$(ls -1d "$CAPTURE_ROOT"/*/ 2>/dev/null | tail -1 || true)
if [ -n "$latest" ] && [ $(( $(date +%s) - $(stat -c %Y "$latest") )) -lt "$MIN_GAP_SECONDS" ]; then
  exit 0
fi

out=$CAPTURE_ROOT/$(date +%Y%m%dT%H%M%S)
mkdir -p "$out"

{
  echo "time: $(date -Is)"
  echo "main_pid: $main_pid"
  for f in memory.current memory.swap.current memory.high memory.max memory.events memory.stat; do
    echo "== $f"; cat "$cg_dir/$f"
  done
} > "$out/cgroup.txt"

ps -o pid,ppid,rss,vsz,etime,args --sort=-rss -p "$(paste -sd, "$cg_dir/cgroup.procs")" > "$out/ps.txt"
for pid in $(cat "$cg_dir/cgroup.procs"); do
  { echo "== $pid $(tr '\0' ' ' < "/proc/$pid/cmdline" | cut -c1-200)"; cat "/proc/$pid/smaps_rollup"; } >> "$out/smaps_rollup.txt" 2>/dev/null || true
done

# Stacks only, no --locals: locals hold request bodies and API keys.
timeout 30 "$PY_SPY" dump --pid "$main_pid" > "$out/py-spy.txt" 2>&1 || true

ss -tnp state established "( sport = :8900 )" > "$out/connections.txt" 2>&1 || true

db_url=$(python3 -I -c "
import re
s = open('$GATEWAY_DIR/gateway.env').read()
m = re.search(r'^DATABASE_URL=[\"\']?([^\"\'\n]+)', s, re.M)
print(m.group(1).split('?')[0] if m else '')")
if [ -n "$db_url" ]; then
  psql "$db_url" -X -A -F $'\t' -c "
    SELECT pid, state, now() - query_start AS running_for, wait_event_type, wait_event, left(query, 600) AS query
    FROM pg_stat_activity
    WHERE datname = current_database() AND pid <> pg_backend_pid() AND state <> 'idle'
    ORDER BY query_start" > "$out/pg_activity.txt" 2>&1 || true
fi

tail -n 300 "$GATEWAY_DIR/logs/litellm.log" > "$out/litellm.log.tail" 2>/dev/null || true

chown -R "$OWNER:$OWNER" "$CAPTURE_ROOT"
logger -t llm-litellm-mem-watch "LiteLLM memory.current=$current >= $THRESHOLD_BYTES, captured $out"
