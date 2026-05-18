# llm-gateway

A lightweight local LLM gateway that dynamically allocates GPU slots for [vLLM](https://github.com/vllm-project/vllm) backends and exposes them through a [LiteLLM](https://github.com/BerriAI/litellm) proxy.

## What it does

```
Client
  │
  ▼
LiteLLM proxy  (:8901)   — OpenAI-compatible API, routing, fallbacks
  │
  ▼
model_manager  (:8002)   — Dynamic GPU slot allocator
  │               │
  ▼               ▼
vLLM slot 0    vLLM slot 1    …  vLLM slot N
GPU 0 :9000    GPU 1 :9010      GPU N :9000+N×10
(on-demand)    (on-demand, scale-out)
```

**model_manager** is the core piece. It sits between LiteLLM and vLLM and handles:

- **Cold start** — first request for a model spins up vLLM on a free GPU slot
- **Idle unload** — no requests for N seconds → kill vLLM, free the slot
- **Scale-out** — ≥2 concurrent requests on an occupied slot → spawn a second instance on the next free slot, round-robin across both
- **Scale-in** — the extra instance idles out naturally
- **gpu_busy 503** — all slots occupied by other models → immediate error (no queue)

## Key design decisions

### Why not just run vLLM directly?

On a 2-GPU workstation where GPU 0 is also used for desktop/other tasks, you want:
- Models that load on demand (not always occupying VRAM)
- Automatic unloading after inactivity
- Multiple models sharing the same GPUs with mutual exclusion

### Port layout

vLLM's EngineCore subprocess binds `api_port + N` (typically +2) for its internal ZMQ IPC socket. Slots must be spaced far enough apart that one slot's EngineCore cannot collide with another slot's API port.

```
Slot 0  API :9000  →  EngineCore IPC :9002  (internal, not exposed)
Slot 1  API :9010  →  EngineCore IPC :9012  (well clear of slot 0)
Slot 2  API :9020  →  EngineCore IPC :9022  …
```

If you assign consecutive ports (e.g. 9000 and 9001), slot 1 will always fail with *Address already in use*. The default gap of 10 is conservative and safe.

### Scale-out threshold

Scale-out only fires when **total concurrent active requests ≥ 2**. A single background health-check from LiteLLM is not enough to trigger a second GPU spawn. This prevents runaway GPU usage for low-load scenarios.

## Requirements

- Python 3.12+
- [vLLM](https://github.com/vllm-project/vllm) (any recent version with v1 engine)
- [LiteLLM](https://github.com/BerriAI/litellm) (optional — model_manager works standalone)
- NVIDIA GPU(s) with sufficient VRAM
- `nvidia-smi`, `ss` (iproute2) — used by `monitor.py`

## Setup

### 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install aiohttp litellm
# vLLM must be installed separately — see https://docs.vllm.ai/en/latest/getting_started/installation.html
```

### 2. Configure GPU slots

**GPU slots are discovered automatically** — no code editing needed for common cases.

By default, model_manager calls `nvidia-smi` at startup to enumerate all available GPUs and assigns:
- Slot 0 → GPU 0 → port 9000
- Slot 1 → GPU 1 → port 9010
- Slot 2 → GPU 2 → port 9020
- …and so on (port gap of 10 between slots)

Override with environment variables if needed:

| Variable | Example | Description |
|---|---|---|
| `GPU_SLOTS` | `0:9000,1:9010` | Explicit `gpu_id:api_port` pairs (overrides all other slot config) |
| `GPU_IDS` | `2,3` | Restrict auto-detection to these GPU indices (skip GPU 0/1 used by desktop) |
| `GPU_PORT_BASE` | `9000` | First slot's API port (default: 9000) |
| `GPU_PORT_GAP` | `10` | Port gap between slots (default: 10; must be > 2 — see port layout above) |

Examples:

```bash
# 4-GPU server, use all GPUs
python3 model_manager.py

# 8-GPU server, reserve GPUs 0-3 for other workloads, use GPUs 4-7
GPU_IDS=4,5,6,7 python3 model_manager.py

# Workstation: GPU 0 reserved for desktop, GPU 1 for models
GPU_IDS=1 python3 model_manager.py

# Fully explicit: GPU 0 on port 9000, GPU 2 on port 9020 (skip GPU 1)
GPU_SLOTS=0:9000,2:9020 python3 model_manager.py
```

Edit `MODEL_CONFIGS` in `model_manager.py` to register your models:

```python
MODEL_CONFIGS: dict[str, tuple[str, str]] = {
    "qwen3.6-35b-a3b": ("run_qwen36_35b.sh", "qwen3.6-35b-a3b"),
    "qwen3.6-27b":     ("run_qwen36_27b.sh",  "qwen3.6-27b"),
}
```

### 3. Write a startup script per model

Each script receives `VLLM_CUDA_DEVICE` and `VLLM_PORT` from model_manager at spawn time. Use them:

```bash
#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=${VLLM_CUDA_DEVICE:-0}

exec vllm serve your-model/name \
  --host 127.0.0.1 \
  --port ${VLLM_PORT:-9002} \
  --served-model-name your-model-name \
  --gpu-memory-utilization 0.93 \
  # ... other vllm flags
```

### 4. Configure LiteLLM (optional)

Edit `config.yaml` — point each model's `api_base` at model_manager (`:8002`):

```yaml
model_list:
  - model_name: your-model-name
    litellm_params:
      model: openai/your-model-name
      api_base: http://127.0.0.1:8002/v1
      api_key: your-vllm-api-key
```

### 5. Run

```bash
# model_manager alone (listens on :8002)
python3 model_manager.py

# with systemd (user service)
cp systemd/llm-model-manager.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now llm-model-manager

# LiteLLM proxy (reads config.yaml, listens on :8901)
litellm --config config.yaml --port 8901
```

## Environment variables

### GPU slot configuration

| Variable | Default | Description |
|---|---|---|
| `GPU_SLOTS` | *(auto)* | Explicit `gpu_id:api_port` pairs, e.g. `0:9000,1:9010` |
| `GPU_IDS` | *(all)* | Comma-separated GPU indices to use, e.g. `1,2,3` |
| `GPU_PORT_BASE` | `9000` | First slot's API port |
| `GPU_PORT_GAP` | `10` | Port gap between slots (must be > 2) |

### Behaviour tuning

| Variable | Default | Description |
|---|---|---|
| `IDLE_TIMEOUT` | `300` | Seconds of inactivity before a slot is unloaded |
| `WAKE_TIMEOUT` | `300` | Max seconds to wait for vLLM to become healthy |
| `HEALTH_POLL`  | `2.0` | Seconds between `/health` polls during startup |
| `LISTEN_PORT`  | `8002` | Port model_manager listens on |

## Monitoring

`monitor.py` runs continuously and checks:

- model-manager service health
- Per-slot vLLM API health (`/health`)
- EngineCore process presence (by parent-PID, not fixed port offset)
- **Zombie EngineCore detection** — an EngineCore whose parent died will hold ports and block the next spawn; the monitor flags it and prints the `kill` command
- GPU VRAM usage per slot
- Idle-slot VRAM leaks
- Port audit on all controlled API ports
- Recent events from `model_manager.log`

```bash
python3 monitor.py          # refresh every 30s
python3 monitor.py --once   # one-shot
INTERVAL=60 python3 monitor.py

# background via tmux
tmux new -d -s gateway-monitor 'cd /path/to/llm-gateway && python3 monitor.py'
```

Example output:

```
──────────────────────────────────────────────────────────────────
  LLM Gateway Self-Check  2026-05-18 15:02:07
──────────────────────────────────────────────────────────────────

SERVICE
  model-manager   ✓ running  PID 2600846  uptime 7m 34s

GPU VRAM
  GPU 0  [██████████████████░░] 92%  29,892/32,607 MiB
  GPU 1  [░░░░░░░░░░░░░░░░░░░░]  0%       2/32,607 MiB

SLOTS

  Slot 0  GPU=0  API port:9002
    vLLM API      ✓ HEALTHY  PID 2603148
    EngineCore    ✓ running  PID 2605857  IPC ports: :9004, ...
    VRAM          [██████████████████░░] 92%  29,892/32,607 MiB

  Slot 1  GPU=1  API port:9010
    vLLM API      idle  (port 9010 free)
    EngineCore    idle
    VRAM          [░░░░░░░░░░░░░░░░░░░░]  0%       2/32,607 MiB

ZOMBIE CHECK
  ✓ no zombie EngineCore processes found

ANOMALIES
  ✓ none — system looks healthy
```

## Responses API

model_manager implements a minimal [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) shim (`POST /v1/responses`) with in-memory conversation history, allowing clients that use the Responses API to talk to vLLM backends.

## Tested with

- vLLM 0.19.0
- Qwen3.6-35B-A3B-NVFP4 (`compressed-tensors` quantization, 2× RTX 5090)
- Qwen3.6-27B-Text-NVFP4-MTP (speculative decoding with MTP)
- LiteLLM 1.83.x

## License

MIT
