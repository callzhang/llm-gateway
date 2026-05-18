#!/usr/bin/env python3
"""
monitor.py — LLM Gateway continuous self-check.

Validates the full system contract:

  1. model-manager service (systemd) is active and /health responds
  2. Each GPU slot's vLLM API server (/health on its API port)
  3. EngineCore processes: detected by parent-PID, not a fixed port offset
     (vLLM picks the EngineCore IPC port dynamically — it is NOT always api+1)
  4. Zombie EngineCore detection: any VLLM::EngineCor whose parent is dead
     will block the next spawn with "Address already in use"
  5. GPU VRAM usage via nvidia-smi
  6. Port conflict on controlled API ports

Run:
  python3 monitor.py           # print report every 30 s
  python3 monitor.py --once    # print once and exit
  INTERVAL=60 python3 monitor.py
"""

import argparse
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime

# ── Configuration (mirrors model_manager.py) ───────────────────────────────────
INTERVAL = int(os.environ.get("INTERVAL", "30"))

# Slots are discovered at startup via the same env vars as model_manager.py.
# Set GPU_SLOTS / GPU_IDS / GPU_PORT_BASE / GPU_PORT_GAP to match your deployment.
# (populated after _discover_slots() is defined below)

MODEL_MANAGER_PORT = int(os.environ.get("LISTEN_PORT", "8002"))
LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "logs", "model_manager.log")


def _discover_slots() -> list[tuple[int, int, int]]:
    """Mirror the GPU slot discovery logic from model_manager.py.

    Reads the same env vars so monitor and model_manager always agree:
      GPU_SLOTS="0:9000,1:9010,2:9020"   explicit gpu:port pairs
      GPU_IDS="0,2"                        restrict auto-detection
      GPU_PORT_BASE=9000                   base port (default 9000)
      GPU_PORT_GAP=10                      port gap per slot (default 10)
    """
    base = int(os.environ.get("GPU_PORT_BASE", "9000"))
    gap  = int(os.environ.get("GPU_PORT_GAP",  "10"))

    if slot_str := os.environ.get("GPU_SLOTS", "").strip():
        slots = []
        for i, token in enumerate(slot_str.split(",")):
            token = token.strip()
            if ":" in token:
                g, p = token.split(":", 1)
                slots.append((i, int(g), int(p)))
            else:
                slots.append((i, int(token), base + i * gap))
        return slots

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        gpu_ids = [int(x) for x in out.splitlines() if x.strip().isdigit()]
    except Exception:
        gpu_ids = [0]

    if ids_str := os.environ.get("GPU_IDS", "").strip():
        wanted = {int(x) for x in ids_str.split(",") if x.strip().isdigit()}
        gpu_ids = [g for g in gpu_ids if g in wanted]

    return [(i, gid, base + i * gap) for i, gid in enumerate(gpu_ids)]


# ── ANSI ───────────────────────────────────────────────────────────────────────
_NO_COLOR = not sys.stdout.isatty()

def _c(code: str, text: str) -> str:
    return text if _NO_COLOR else f"{code}{text}\033[0m"

def green(t):  return _c("\033[92m", t)
def yellow(t): return _c("\033[93m", t)
def red(t):    return _c("\033[91m", t)
def cyan(t):   return _c("\033[96m", t)
def bold(t):   return _c("\033[1m",  t)
def dim(t):    return _c("\033[2m",  t)

def tick(s): return green(f"✓ {s}")
def warn(s): return yellow(f"⚠ {s}")
def fail(s): return red(f"✗ {s}")


# ── Subprocess / HTTP helpers ──────────────────────────────────────────────────

def sh(cmd: list[str], timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def http_ok(url: str, timeout: int = 3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


# ── Process helpers ────────────────────────────────────────────────────────────

def all_processes() -> list[dict]:
    """Return list of {pid, ppid, comm} for every running process."""
    out = sh(["ps", "-e", "-o", "pid=,ppid=,comm=", "--no-headers"])
    result = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and parts[0].isdigit():
            result.append({
                "pid":  int(parts[0]),
                "ppid": int(parts[1]) if parts[1].isdigit() else 0,
                "comm": parts[2].strip(),
            })
    return result


def port_owner(port: int) -> dict | None:
    """Return {pid, ppid, comm} of the process listening on *port*, or None."""
    out = sh(["ss", "-tlnpH", f"sport = :{port}"])
    m = re.search(r'pid=(\d+)', out)
    if not m:
        return None
    pid = int(m.group(1))
    ps_out = sh(["ps", "-o", "ppid=,comm=", "-p", str(pid)])
    parts = ps_out.split(None, 1)
    ppid = int(parts[0]) if parts and parts[0].strip().lstrip('-').isdigit() else 0
    comm = parts[1].strip() if len(parts) > 1 else "?"
    return {"pid": pid, "ppid": ppid, "comm": comm}


def children_of(ppid: int, procs: list[dict]) -> list[dict]:
    return [p for p in procs if p["ppid"] == ppid]


def enginecore_ports_of(ec_pid: int) -> list[int]:
    """Return all TCP ports this EngineCore is listening on."""
    out = sh(["ss", "-tlnpH"])
    ports = []
    for line in out.splitlines():
        if f"pid={ec_pid}" in line:
            # Extract port from address like 127.0.0.1:9011 or *:9011
            m = re.search(r':(\d+)\s', line)
            if m:
                ports.append(int(m.group(1)))
    return sorted(ports)


def gpu_stats() -> dict[int, dict]:
    out = sh(["nvidia-smi",
              "--query-gpu=index,memory.used,memory.total,memory.free",
              "--format=csv,noheader,nounits"])
    result = {}
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 4 and p[0].isdigit():
            result[int(p[0])] = {
                "used": int(p[1]), "total": int(p[2]), "free": int(p[3])
            }
    return result


def systemd_service_state() -> dict:
    out = sh(["systemctl", "--user", "show", "llm-model-manager",
              "--property=ActiveState,MainPID,ExecMainStartTimestamp"])
    active_m = re.search(r"ActiveState=(\S+)", out)
    pid_m    = re.search(r"MainPID=(\d+)", out)
    ts_m     = re.search(r"ExecMainStartTimestamp=(.+)", out)
    return {
        "active": active_m.group(1) if active_m else "unknown",
        "pid":    int(pid_m.group(1)) if pid_m else 0,
        "since":  ts_m.group(1).strip() if ts_m else "",
    }


def uptime_str(since: str) -> str:
    try:
        parts = since.split()
        if len(parts) >= 3:
            ts    = datetime.strptime(" ".join(parts[1:3]), "%Y-%m-%d %H:%M:%S")
            delta = max(0, int(time.time() - ts.timestamp()))
            h, rem = divmod(delta, 3600)
            m, s   = divmod(rem, 60)
            if h:  return f"{h}h {m}m"
            if m:  return f"{m}m {s}s"
            return f"{s}s"
    except Exception:
        pass
    return ""


def vram_bar(used: int, total: int, width: int = 20) -> str:
    pct    = used / total if total else 0
    filled = round(pct * width)
    bar    = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct*100:.0f}%  {used:,}/{total:,} MiB"


def last_log_lines(n: int = 6) -> list[str]:
    try:
        raw = sh(["tail", "-100", LOG_PATH])
        keep = [l for l in raw.splitlines()
                if any(kw in l for kw in (
                    "ready", "Scale", "scale", "failed", "cool",
                    "claimed", "Spawn", "unload", "ERROR", "WARNING",
                    "complete",
                ))
                and "aiohttp.access" not in l]
        return keep[-n:]
    except Exception:
        return []


# ── Main report ────────────────────────────────────────────────────────────────

def report() -> str:
    now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out: list[str] = []
    anomalies: list[str] = []

    sep = "─" * 66
    out.append(f"\n{bold(sep)}")
    out.append(f"{bold('  LLM Gateway Self-Check')}  {dim(now)}")
    out.append(bold(sep))

    # ── 1. model-manager ──────────────────────────────────────────────────────
    out.append(f"\n{bold('SERVICE')}")
    svc        = systemd_service_state()
    mm_healthy = http_ok(f"http://127.0.0.1:{MODEL_MANAGER_PORT}/health")
    uptime     = uptime_str(svc["since"])

    if svc["active"] == "active" and mm_healthy:
        out.append(f"  model-manager   {tick('running')}  "
                   f"PID {svc['pid']}  uptime {uptime}")
    elif svc["active"] == "active":
        out.append(f"  model-manager   {warn('active but /health failed')}  "
                   f"PID {svc['pid']}")
        anomalies.append("model-manager is active but /health returned non-200")
    else:
        out.append(f"  model-manager   {fail(svc['active'])}")
        anomalies.append(
            f"model-manager service is {svc['active']!r}  "
            "→  systemctl --user restart llm-model-manager")

    # ── 2. GPU VRAM overview ─────────────────────────────────────────────────
    out.append(f"\n{bold('GPU VRAM')}")
    gpu = gpu_stats()
    for idx in sorted(gpu):
        g = gpu[idx]
        out.append(f"  GPU {idx}  {vram_bar(g['used'], g['total'])}")

    # ── 3. Slot health (uses process tree, not fixed port offsets) ────────────
    out.append(f"\n{bold('SLOTS')}")
    procs = all_processes()  # snapshot of all running processes

    # Build map: api_port → (api_pid, [EngineCore children])
    running_api_pids: dict[int, int] = {}   # api_port → api_pid
    slot_enginecores: dict[int, list[dict]] = {}  # slot_id → [{pid, ports}]

    for sid, gpu_id, api_port in SLOTS:
        api_proc = port_owner(api_port)
        if api_proc:
            api_pid = api_proc["pid"]
            running_api_pids[api_port] = api_pid
            # Find EngineCore children
            ec_children = [
                p for p in children_of(api_pid, procs)
                if "EngineCor" in p["comm"]
            ]
            slot_enginecores[sid] = [
                {"pid": p["pid"], "ports": enginecore_ports_of(p["pid"])}
                for p in ec_children
            ]

    for sid, gpu_id, api_port in SLOTS:
        out.append(f"\n  {bold(f'Slot {sid}')}  GPU={gpu_id}  API port:{api_port}")
        api_proc = port_owner(api_port)

        if api_proc:
            healthy = http_ok(f"http://127.0.0.1:{api_port}/health")
            hstr    = tick("HEALTHY") if healthy else warn("UNHEALTHY")
            out.append(f"    vLLM API      {hstr}  PID {api_proc['pid']}")
            if not healthy:
                anomalies.append(
                    f"Slot {sid} vLLM API (:{api_port}) up but /health failed")

            ecs = slot_enginecores.get(sid, [])
            if ecs:
                for ec in ecs:
                    ports_str = ", ".join(f":{p}" for p in ec["ports"]) or "(no TCP)"
                    out.append(f"    EngineCore    {tick('running')}  "
                               f"PID {ec['pid']}  IPC ports: {ports_str}")
            else:
                out.append(f"    EngineCore    {dim('not yet visible')}  "
                           "(still initialising or uses Unix sockets)")
        else:
            out.append(f"    vLLM API      {dim('idle')}  (port {api_port} free)")
            out.append(f"    EngineCore    {dim('idle')}")

        g = gpu.get(gpu_id)
        if g:
            out.append(f"    VRAM          {vram_bar(g['used'], g['total'])}")
            # Warn if idle slot has suspiciously high VRAM usage
            if not api_proc and g["used"] > 500:
                anomalies.append(
                    f"GPU {gpu_id} shows {g['used']:,} MiB used but slot {sid} is idle "
                    "— possible leaked CUDA context")

    # ── 4. Zombie EngineCore detection ────────────────────────────────────────
    # Any VLLM::EngineCor process whose parent is NOT in our running API server
    # set is a zombie. It will hold ports and block the next spawn.
    out.append(f"\n{bold('ZOMBIE CHECK')}")
    all_api_pids = set(running_api_pids.values())
    zombies = [
        p for p in procs
        if "EngineCor" in p["comm"] and p["ppid"] not in all_api_pids
    ]
    if zombies:
        for z in zombies:
            held = enginecore_ports_of(z["pid"])
            ports_str = ", ".join(f":{p}" for p in held) or "(no TCP)"
            line = (f"  {fail('ZOMBIE EngineCore')}  PID {z['pid']}  "
                    f"PPID {z['ppid']} (dead)  ports: {ports_str}")
            out.append(line)
            anomalies.append(
                f"Zombie EngineCore PID {z['pid']} (parent {z['ppid']} is dead) "
                f"holds ports {ports_str}. "
                f"Next spawn will fail.  Fix: kill {z['pid']}")
    else:
        out.append(f"  {tick('no zombie EngineCore processes found')}")

    # ── 5. API port audit ─────────────────────────────────────────────────────
    out.append(f"\n{bold('PORT AUDIT')}  (controlled API ports only)")
    for sid, gpu_id, api_port in SLOTS:
        proc = port_owner(api_port)
        if proc:
            out.append(f"  :{api_port}  Slot {sid} API   "
                       f"{cyan(proc['comm'])} PID {proc['pid']}")
        else:
            out.append(f"  :{api_port}  Slot {sid} API   {dim('free')}")

    # Check: is any API port held by the wrong process (not a vllm)?
    for sid, gpu_id, api_port in SLOTS:
        proc = port_owner(api_port)
        if proc and "vllm" not in proc["comm"].lower():
            anomalies.append(
                f"Slot {sid} API port {api_port} is held by "
                f"'{proc['comm']}' (PID {proc['pid']}) — not a vLLM process! "
                "Spawns for this slot will fail.")

    # ── 6. Recent log events ──────────────────────────────────────────────────
    out.append(f"\n{bold('RECENT EVENTS')}")
    for line in last_log_lines(6):
        if any(w in line for w in ("ERROR", "failed", "ZOMBIE")):
            out.append(f"  {red(line)}")
        elif any(w in line for w in ("WARNING", "cool", "warn")):
            out.append(f"  {yellow(line)}")
        elif any(w in line for w in ("ready", "complete", "INFO")):
            out.append(f"  {dim(line)}")
        else:
            out.append(f"  {line}")

    # ── 7. Anomaly summary ────────────────────────────────────────────────────
    out.append(f"\n{bold('ANOMALIES')}")
    if anomalies:
        for a in anomalies:
            out.append(f"  {fail(a)}")
    else:
        out.append(f"  {tick('none — system looks healthy')}")

    out.append(f"\n{dim(f'Interval: {INTERVAL}s   Ctrl-C to stop')}")
    return "\n".join(out)


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    global SLOTS
    SLOTS = _discover_slots()

    parser = argparse.ArgumentParser(description="LLM Gateway monitor")
    parser.add_argument("--once", action="store_true",
                        help="Print one report and exit")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        while True:
            print(report(), flush=True)
            if args.once:
                break
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
