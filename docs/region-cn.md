# CN region: a second gateway stack on the A100 host

Two independent stacks serve the same public contract (one virtual key, one model
list). Each region serves what it can locally and borrows the other region's GPUs
only when its own queue is long.

| | US region (GPU4, `stardust-gpu4`) | CN region (A100, `a10041` / Tailscale `a100-1`) |
|---|---|---|
| GPUs | 2 × RTX 5090 32 GB | 7 × A100 40 GB, **shared** with other teams |
| Models served locally | all (27b, 35b heretic, TTS, ASR) | `qwen3.8-27b` only |
| Weights | NVFP4 (native FP4) | same NVFP4 checkpoint and revision, Marlin W4A16 kernel |
| Checkout | `/home/derek/Projects/llm-gateway` | `/data/derek/llm-gateway` (mode 700) |
| LiteLLM config | `config.yaml` | `config.cn.yaml` (`LITELLM_CONFIG`) |
| model_manager | `127.0.0.1` + tailnet `:8002` | Tailscale IP only `100.118.88.28:8002` |
| Database | Postgres `127.0.0.1:5433` (owner) | the same database over `llm-pg-peer.socket`, role `litellm_cn` |
| Public entrance | `llm-api.preseen.ai` (tunnel `preseen-gateway`) | `llm-api-geo.preseen.ai` (tunnel `llm-api-geo`, CN replica only) |
| Units | `systemd/*.service` (user), `systemd/system/llm-pg-peer.*` (root) | `systemd/region-cn/*` (user, linger on) |

## Request path

```
client ─► Cloudflare ─► tunnel ─► LiteLLM :8900 ──order 1──► local model_manager :8002 ─► vLLM
                                     │
                                     └─order 2 (only after a 503 queue_overflow)─► peer model_manager :8002
                                        (Tailscale, header x-gateway-overflow: 1)
```

* `OVERFLOW_QUEUE_DEPTH=4` (both `gateway.env`): once a model's summed vLLM queue reaches 4,
  model_manager answers new chat work with `503 queue_overflow`. Sticky hits stay local (prefix
  cache). A request carrying `x-gateway-overflow` already crossed once and is always served, so
  two saturated regions cannot bounce it.
* LiteLLM 1.99 tries `order: 2` only after `order: 1` fails. `disable_cooldowns: true` keeps a
  burst of overflow 503s from benching the local deployment.
* CN forwards every model other than the 27B straight to GPU4's model_manager.
* Both LiteLLMs share one Postgres, so keys, budgets, spend logs and `previous_response_id`
  work from either entrance. Only the US LiteLLM migrates the schema
  (`disable_prisma_schema_update` in `config.cn.yaml`) and runs the 7-day spend-log cleanup.

## Network facts (measured 2026-11-07)

* GPU4 ↔ A100 has no direct Tailscale path: both sides are behind NAT that changes the port per
  destination (`MappingVariesByDestIP: true`), with no port mapping, so traffic goes through the
  Tailscale relay in Nuremberg. RTT is 213–537 ms; a health check takes about 1 s. Overflow adds
  that latency to the first token. A UDP port forward on either side's router would allow a direct
  path (the US↔China physical floor is still about 150 ms).
* **Cloudflare replicas do not split traffic by region here.** The A100's cloudflared registers at
  Cloudflare LAX, and A100-originated requests reach Cloudflare at NRT (`loc=JP`). With one replica
  per region on the same tunnel, US requests (SJC) went 12/12 to GPU4, but requests from the A100
  split 5 to GPU4 and 7 to the A100. So `llm-api.preseen.ai` stays on `preseen-gateway` (GPU4 only),
  and `llm-api-geo` runs only the CN replica. Deterministic splitting by visitor country needs
  Cloudflare Load Balancing geo steering (a paid add-on), or callers in China use
  `llm-api-geo.preseen.ai` explicitly.

## A100 specifics

* The system `nvcc` is 11.5, too old for FlashInfer's JIT on CUDA 13. The vLLM venv carries
  `nvidia-cuda-nvcc==13.0.*` (+ runtime, cccl, crt, nvvm) and `ninja`; `MODEL_CUDA_HOME` points
  at `site-packages/nvidia/cu13`, with `lib64 -> lib` and `libcudart.so -> libcudart.so.13` symlinks.
* The root disk is 96% full: weights, venvs, the uv cache and logs all live on `/data/derek`.
  `/data` has a default ACL of `other::rwx`, so umask does not protect new files there. The checkout
  is mode 700 with its ACLs stripped; keep secrets inside it.
* Ports: slots start at 9300 (`GPU_PORT_BASE`) and co-resident lanes at 9500, because 9027 and 9100
  are taken on this host.
* Six other users have sudo on this host. They can read `gateway.env` (LiteLLM master key, the
  `litellm_cn` password) and, through the database, every virtual key hash and seven days of stored
  prompts. Derek accepted this on 2026-11-07. `/admin/kill|start|switch` are loopback-only, so
  they cannot drive GPU4's slots over Tailscale.

## Single-card measurements (GPU 3, A100-PCIE-40GB, random 2048 in / 512 out)

| config | conc. | out tok/s | total tok/s | TPOT ms | GPU util | power | SM clock |
|---|---|---|---|---|---|---|---|
| MTP, seqs 16 | 1 | 74.6 | 381 | 11.7 | 91% | 247 W | 1318 MHz |
| MTP, seqs 64, mbt 8192 | 16 | 339.5 | 1732 | 39.6 | 97% | 247 W | 1156 MHz |
| MTP, seqs 64, mbt 8192 | 64 | 370.8 | 1892 | 88.2 | 99% | 249 W | 1148 MHz |
| no MTP, seqs 64, mbt 8192 | 1 | 54.9 | 280 | 16.6 | 97% | 246 W | 1285 MHz |
| no MTP, seqs 64, mbt 8192 | 16 | 347.4 | 1772 | 29.0 | 99% | 249 W | 1171 MHz |
| no MTP, seqs 64, mbt 8192 | 64 | 437.9 | 2234 | 117.3 | 100% | 249 W | 1138 MHz |

KV cache at util 0.90 / 131072 context: 414–497k tokens (5090: 254,862). VRAM peaks at 40.0 of
40.96 GB. MTP mean acceptance length is 1.70–1.76.

**What limits it.** The PCIe A100 is capped at 250 W, and every configuration, even a single stream,
draws 246–249 W with the SM clock throttled from 1410 to 1138–1318 MHz. The card is at its power
limit, not idle. Single-stream decode without MTP moves about 15.4 GB of weights per 16.6 ms, which
is about 930 GB/s, or 60% of the 1555 GB/s HBM peak. NVFP4 on Ampere is dequantized in software by
Marlin (FP4 values with FP8 scales, and no FP8 hardware), which costs ALU work and power for every
weight byte. That is why the card hits the power cap before it reaches the bandwidth roofline.
Getting closer would mean a format Ampere dequantizes cheaply (AWQ/GPTQ INT4 Marlin), which gives
up byte-identical weights with GPU4.

MTP stays on: it makes single-stream decode 1.43× faster and is equal at 16 concurrent; it only
loses at 64 concurrent (−15%), which the 8-seq slot limit never reaches.

## Operations

```bash
ssh a100-1 'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user status llm-model-manager llm-litellm llm-cloudflared'
```

Deploying to the A100: `git archive HEAD | ssh a100-1 'tar -x -C /data/derek/llm-gateway'`, then
restart the unit. model_manager keeps and re-adopts its vLLM children across a restart, as on GPU4.
The A100 `gateway.env` is not tracked. Its variables are `GPU_PORT_BASE`,
`GPU_CORESIDENT_PORT_BASE`, `LISTEN_HOST`, `PYTHON_BIN`, `VLLM_BIN`, `MODEL_HF_HOME`,
`MODEL_CUDA_HOME`, `PATH`, `MAX_JOBS`, `WAKE_TIMEOUT`, `OVERFLOW_QUEUE_DEPTH`, `HF_HOME`,
`HF_HUB_OFFLINE`, `LITELLM_CONFIG`, `LOCAL_GATEWAY_API_BASE`, `PEER_GATEWAY_API_BASE`,
`DATABASE_URL`, `DISABLE_SCHEMA_UPDATE`, `LITELLM_MASTER_KEY`, `GATEWAY_PY_BIN` and
`PRISMA_BINARY_CACHE_DIR`.

Rollback of overflow on GPU4: remove `OVERFLOW_QUEUE_DEPTH` and `PEER_GATEWAY_API_BASE` from
`gateway.env` and the order-2 entry from `config.yaml`, then restart `llm-litellm` and
`llm-model-manager`. The CN stack can stay up or be stopped with
`systemctl --user stop llm-gateway.target` on the A100.
