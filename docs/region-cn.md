# CN region: a second gateway stack on the A100 host

Two independent stacks serve the same public contract (one virtual key, one model
list). Each region serves what it can locally and borrows the other region's GPUs
only when its own queue is long.

| | US region (GPU4, `stardust-gpu4`) | CN region (A100, `a10041` / Tailscale `a100-1`) |
|---|---|---|
| GPUs | 2 × RTX 5090 32 GB | 7 × A100 40 GB, **shared** with other teams |
| Models served locally | all (27b, 35b heretic, TTS, ASR) | `qwen3.8-27b`, TTS, ASR (35B goes to GPU4) |
| Weights | NVFP4 (native FP4) | same NVFP4 checkpoint and revision, Marlin W4A16 kernel |
| Checkout | `/home/derek/Projects/llm-gateway` | `/data/derek/llm-gateway` (mode 700) |
| LiteLLM config | `config.yaml` | `config.cn.yaml` (`LITELLM_CONFIG`) |
| model_manager | `127.0.0.1` + tailnet `:8002` | Tailscale IP only `100.118.88.28:8002` |
| Database | Postgres `127.0.0.1:5433` (owner) | the same database over `llm-pg-peer.socket`, role `litellm_cn` |
| Public entrance | `llm-api.preseen.ai` (tunnel `preseen-gateway`) | `https://llm-cn.preseen.ai:18900` direct (Caddy TLS, virtual-key auth, open to all sources); `llm-api-geo.preseen.ai` (tunnel, slower, see below) |
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
* TTS and ASR run in both regions too, each region's own first. TTS needs a whole card, so a
  region with no free slot answers `503 gpu_busy` and the other region takes it. ASR has no queue
  to measure: an upload of `ASR_OVERFLOW_MIN_SECONDS` (default 300) or more that would run on the
  CPU worker because no GPU has room gets `503 queue_overflow` instead; shorter clips stay on the
  local CPU, which beats a crossing plus the peer lane's possible ~40 s cold start.
* LiteLLM 1.99 drops a deployment's `extra_headers` on `/audio/speech` and
  `/audio/transcriptions` (verified with a stub backend), so model_manager also treats any request
  from another host (not loopback, not `LISTEN_HOST`) as already crossed.
* CN forwards the 35B straight to GPU4's model_manager.
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

* **Cloudflare is the slow way into the A100.** The A100's overseas traffic leaves through a VPN
  in Tokyo (egress `18.179.246.86`; domestic egress is the site NAT `27.221.10.54`), and its
  tunnel connections register at LAX. Measured from the A100 on 2026-11-07, time to first byte
  of `/health/liveliness` was 0.79–0.83 s via `llm-api` (to GPU4) and 0.96–1.13 s via
  `llm-api-geo` (to the A100 itself). Geo steering on `llm-api` would make CN callers slower.

## Direct entrance for CN services

CN services call the A100 without Cloudflare or a border crossing:

```
CN service ─► https://llm-cn.preseen.ai:18900 ─► site NAT 27.221.10.54:18900 ─► A100 172.20.10.41:18900
             (Caddy: TLS, CN_ALLOW_CIDRS) ─► LiteLLM 127.0.0.1:8900 ─► same routing as above
```

* LiteLLM on the A100 listens on loopback (`LITELLM_HOST=127.0.0.1`), because the host has no
  firewall and other users share its LAN. Caddy (`systemd/region-cn/Caddyfile`, unit
  `llm-tls-proxy`) is the only network-facing gateway port. It is open to every source
  (`CN_ALLOW_CIDRS="0.0.0.0/0 ::/0"`, Derek 2026-11-08: user IPs are not restricted); the
  virtual key is the access control. A narrower `CN_ALLOW_CIDRS` gives other sources a 403.
* The certificate (Let's Encrypt, EC256) is issued on GPU4 by DNS-01 with lego and pushed by
  `sudo scripts/region_cn_tls.sh renew`, which is a no-op until 30 days before expiry. The daily
  `llm-cn-tls-renew.timer` (`systemd/system`) runs it. It is issued on GPU4
  so the zone-wide Cloudflare token never reaches the shared host.
* Done: the `llm-cn.preseen.ai` A record (DNS only), the first certificate (valid to
  2027-02-05), and the renewal timer on GPU4 (installed 2026-11-08; a manual run succeeded).
* Still to do: the NAT port forward `27.221.10.54:18900 -> 172.20.10.41:18900` (site network
  admin). `27.221.10.54` is the whole site's shared address, not the A100's: from outside,
  `:18900` answers "connection refused" (2026-11-08, probed from GPU4) while `:443` reaches
  another team's nginx. Everything behind the NAT is verified from inside the LAN.

## A100 specifics

* The system `nvcc` is 11.5, too old for FlashInfer's JIT on CUDA 13. The vLLM venv carries
  `nvidia-cuda-nvcc==13.0.*` (+ runtime, cccl, crt, nvvm) and `ninja`; `MODEL_CUDA_HOME` points
  at `site-packages/nvidia/cu13`, with `lib64 -> lib` and `libcudart.so -> libcudart.so.13` symlinks.
* The root disk is 96% full: weights, venvs, the uv cache and logs all live on `/data/derek`.
  `/data` has a default ACL of `other::rwx`, so umask does not protect new files there. The checkout
  is mode 700 with its ACLs stripped; keep secrets inside it.
* Audio: TTS runs from its own vllm-omni venv `/data/derek/venv-tts` (built with uv from GPU4's
  `llm-gateway-tts` package set, setuptools unpinned). ASR uses `venv-vllm029`, which got the
  `vllm[audio]` extras (av, scipy, soundfile, soxr). Weights were downloaded on the A100 at GPU4's
  revisions; HF_HUB_OFFLINE needs a `refs/main` in each repo, written by hand. Measured: TTS cold
  start 317 s the first time (torch.compile, no cache yet), 31.2 GiB on a card, ~2 s per sentence
  warm; ASR cold start ~38 s, 0.5 s for a short clip, 282 s of speech in 21.8 s.
* vLLM 0.29 picks a CUTLASS FP8 kernel for ASR's online FP8 on Ampere, which has no FP8; the
  first forward fails in `cutlass_scaled_mm_sm80_epilogue`. `ASR_VLLM_DISABLED_KERNELS` lists
  the kernels ahead of Marlin so it falls through to Marlin W8A16.
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

## Multi-GPU scale-out (2026-11-07, 64 closed-loop clients, 2k-token prompts, 512 out, 25 min)

| replicas | GPUs | out tok/s (30 s windows) |
|---|---|---|
| 1 | 3 | 270–340 |
| 2 | 3, 5 | 580–666 |
| 3 | 3, 5, 6 | 887–973 |
| 4 | 3, 4, 5, 6 | 1195–1297 (steady ~1240) |

3050 requests, 0 failures. Each new replica was ready 74 s after its spawn (warm compile cache).
With four replicas, all four cards averaged 97% utilization, 246–250 W (the cap) and 38–39 GB of VRAM.
GPU 4 shares the card with a colleague's mmpose and ComfyUI (2.6 GB). model_manager sized that replica
to util 0.904 from the 36.9 GiB that was free.

A fifth replica on GPU 1 (29.5 GiB free next to NeoHorse) was refused: the 27B's minimum was a
utilization fraction (0.78) calibrated on the 32 GB 5090, which is 31.2 GiB on a 40 GB card even
though the model's real footprint did not grow. The minimum is now in GiB (`MODEL_MIN_VIABLE_GIB`,
24.83 GiB for the 27B) and is converted to a utilization from each card's total, so on the A100 it
is 0.621 and GPU 1 qualifies (needs 25.1 GiB free with the margin; 5 of 7 cards). On the 5090 the
GiB values give back exactly the old 0.84 / 0.78 / 0.095. GPU 0 and GPU 2 are full with other
teams' services.

MTP stays on: it makes single-stream decode 1.43× faster and is equal at 16 concurrent; it only
loses at 64 concurrent (−15%), which the 8-seq slot limit never reaches.

## Operations

```bash
ssh a100-1 'XDG_RUNTIME_DIR=/run/user/$(id -u) systemctl --user status llm-model-manager llm-litellm llm-cloudflared'
```

Deploying to the A100: `git archive HEAD | ssh a100-1 'tar -x -C /data/derek/llm-gateway'`, then
restart the unit. model_manager keeps and re-adopts its vLLM children across a restart, as on GPU4.
After a deploy that changes `patches/qwen3_engine_tool_parser.py`, run
`VLLM_BIN=/data/derek/venv-vllm029/bin/vllm scripts/install_qwen_parser.sh` there (see
`docs/qwen-native-output-grammar-fix.md`). The A100 `gateway.env` is not tracked. Its variables are `GPU_PORT_BASE`,
`GPU_CORESIDENT_PORT_BASE`, `LISTEN_HOST`, `PYTHON_BIN`, `VLLM_BIN`, `MODEL_HF_HOME`,
`MODEL_CUDA_HOME`, `PATH`, `MAX_JOBS`, `WAKE_TIMEOUT`, `OVERFLOW_QUEUE_DEPTH`, `HF_HOME`,
`HF_HUB_OFFLINE`, `LITELLM_CONFIG`, `LOCAL_GATEWAY_API_BASE`, `PEER_GATEWAY_API_BASE`,
`DATABASE_URL`, `DISABLE_SCHEMA_UPDATE`, `LITELLM_MASTER_KEY`, `GATEWAY_PY_BIN`,
`PRISMA_BINARY_CACHE_DIR`, `LITELLM_HOST`, `CN_TLS_HOST`, `CN_TLS_PORT`, `CN_ALLOW_CIDRS`,
`VLLM_ASR_BIN`, `VLLM_TTS_BIN`, `ASR_CPU_PYTHON`, `ASR_HF_HOME`, `ASR_VLLM_DISABLED_KERNELS`
and `QWEN3_TTS_MIN_FREE_GIB` (32.2).

Rollback of overflow on GPU4: remove `OVERFLOW_QUEUE_DEPTH` and `PEER_GATEWAY_API_BASE` from
`gateway.env` and the order-2 entry from `config.yaml`, then restart `llm-litellm` and
`llm-model-manager`. The CN stack can stay up or be stopped with
`systemctl --user stop llm-gateway.target` on the A100.
