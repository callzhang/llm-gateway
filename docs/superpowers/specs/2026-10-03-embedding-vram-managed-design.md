# Embedding provider under model_manager VRAM management — design

Status: **draft, nothing implemented, no cutover scheduled.**
Date: 2026-10-03

## Goal

Bring jina-v5-small (the US `embedding-provider`, `embed.preseen.ai`) under
model_manager's VRAM management so it stops being an unmanaged GPU neighbour.

## Hard constraints (from the memory-connector side)

1. Vectors must stay in the same space: for every item in a sample that includes
   multi-line and >2k-token texts, `cosine(new, current embed.preseen.ai) >= 0.9999`.
   Today: transformers bf16, `trust_remote_code`, snapshot `dd76d535`,
   `EMBEDDING_TASK=text-matching`, never passes `prompt_name` so `encode()`
   prepends `"Document: "` to every input, last-token pooling, L2 normalize,
   max_length 8192, dimensions 1024. The CN vLLM setup (retrieval adapter, no
   prefix) measured cosine 0.59 raw / 0.886 with task=retrieval, so it must not
   be copied.
2. `embed.preseen.ai`, its API key and the model id string stay unchanged.
   memory-connector stamps every vector with
   `profile:provider:base_url:model:dim`; changing any part makes all ~200k stored
   vectors count as stale.
3. Embedder config is never changed on the live checkout; it goes through
   memory-connector main.
4. No new Python envs on gpu4. llm-gateway runs as `derek`, embedding-provider as
   `stardust`.
5. Heads-up to the memory-connector session before any cutover, so it can run
   `GET /v1/console/admin/runtime/embedder/reembed-preview` (expect total 0).

## Measured facts (2026-10-03)

Traffic, last 100k inputs (~14 h, `/request-logz`; lengths only):

| metric | value |
|---|---|
| input chars | p50 12, p90 99, p99 294, p99.9 1,200, max 30,947 |
| inputs > 2,000 chars | 26 / 100,000 |
| texts per request | p50 3, p99 7, p99.9 64, max 660 |

The provider already runs on **CPU for normal traffic** (`START_DEVICE=cpu`),
switches to GPU only when a batch has >= 8 texts, and offloads after 300 s idle
(60 reloads so far). Normal load is ~0.24 req/s. Bursts come from batch writes
and re-embeds.

GPU footprint (bf16, LoRA unmerged, GPU1 with 2.6 GiB free):

| stage | process MiB |
|---|---|
| CUDA context only | 612 |
| weights loaded (1,445 MiB) | 2,078 |
| 1 short text | 2,090 |
| 8 x 300 chars | 2,336 |
| 64 x 150 chars | OOM at ~2.55 GiB |

Activations are roughly 0.15 MiB per token (rough estimate from the 8 x 300 step).
Floor to hold the model idle: **2.1 GiB**; to serve p99 batches: **~2.4 GiB**;
big batches / long texts need ~1 GiB more.

## Agreed VRAM policy

1. On GPU load, take the **maximum currently free VRAM** (no standing reserve),
   not a computed `batch x 8192` cap — real traffic is heavy-tailed, so oversized
   requests are handled by splitting batches, not by reserving memory.
2. When another model managed by this service starts and is **not** a >10B
   primary, the embedding service **yields**: it shrinks its footprint so the
   newcomer can start at its minimum. >10B primaries keep the existing rule
   (`reclaim_coresident_for`: stop idle co-resident models).
3. "CPU-first, GPU on demand" is kept for both implementation options below.

Yield order between small neighbours (decided from the measurements below):
embedding first (CPU path covers normal traffic), then ASR (also has a CPU path),
TTS never (primary-sized, ~29 GiB).

### Why the 27B util is not lowered statically (2026-10-03)

Neighbours (gliner, embedding, ASR) all have CPU paths and are usually on CPU, so
a standing reserve only buys their GPU speed. Lowering 27B util statically would
cost both replicas ~4 GiB permanently (0.01 util = ~326 MiB; fitting embedding +
ASR + gliner at once needs ~7 GiB vs the 3 GiB reserve today, i.e. util ~0.76,
below the 0.78 floor). 27B KV usage from the vLLM logs (sampled only while busy):

| log | samples | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| slot0 (today) | 859 | 52.8% | 82.2% | 96.3% | 99.4% |
| slot0 (yesterday) | 6,876 | 47.9% | 60.7% | 70.6% | 97.5% |
| slot1 (today) | 858 | 52.8% | 68.7% | 73.0% | 90.2% |

Average has slack, peak does not: cutting ~half the KV would cause preemption at
peaks. So the 27B keeps its util and neighbours yield dynamically instead.

### ASR: CPU vs GPU (50 transcriptions from model_manager.log)

| audio length | CPU median / RTF | GPU median / RTF |
|---|---|---|
| <= 30 s | 5.8 s / 0.53 (n=28) | 18.5 s / 2.04 (n=8, includes cold start) |
| 30-300 s | 29.5 s / 0.21 (n=3) | 4.2 s / 0.09 (n=7) |
| > 300 s | 262.8 s / 0.13 (n=1) | 13.9 s / 0.03 (n=3) |

Small samples, orders of magnitude only. GPU pays off only for audio longer than
~30 s, so ASR GPU admission can require a long clip and does not justify taking
KV from the 27B. "No GPU has room" appears 31 times in the model_manager logs.

## Options

### A. Keep the transformers provider; manager does admission and device control (recommended)

- New `request_kind="embedding"` in `ModelConfig`; manager reverse-proxies
  `/v1/embeddings` to the existing provider on :7997 and keeps `embed.preseen.ai`
  pointing at the same upstream chain.
- The provider already exposes `POST /admin/device` (API-key auth, 409 on the
  device-switch cooldown) and `/statsz` (`loaded_device`, `engine_state`). The
  manager yields by telling it to go to CPU and admits it back to GPU only after
  checking free VRAM — **no process spawn, no restart, no cross-user spawn, no
  cold start**, and the stardust unit keeps its identity.
- One engine for CPU and GPU: no parity risk beyond what already exists between
  the provider's CPU and GPU paths.
- Needs: manager-side VRAM admission for a non-vLLM process (GPU usage of that
  pid, not `_gpu_vllm_used_mib`); provider-side change so the GPU load takes a
  VRAM budget and the CPU fallback is **not sticky** (today it stays on CPU after
  one failed GPU load until restart); batch splitting driven by the budget.
- Trade-off: touches the provider repo (stardust-owned) and model_manager core.

### B. vLLM pooling runtime for the GPU path

- Merge the text-matching LoRA into the weights; the caller (or a proxy shim) adds
  `"Document: "`; serve with the pooling runner. Reuses the existing co-resident
  spawn / util / lane machinery unchanged.
- Yielding means stop and restart with a smaller `--gpu-memory-utilization`
  (vLLM cannot resize in place; sleep mode conflicts with `expandable_segments`).
- "CPU-first" still needs the transformers CPU path, so B is really **two engines**
  (vLLM on GPU, transformers on CPU) whose outputs must both pass the 0.9999 gate.
- FP4 is rejected: weights are only ~1.4 GiB so it saves < 1 GiB, and it moves
  cosine to ~0.99x, which forces a full re-embed of ~200k vectors.
- Trade-off: smallest manager change, but biggest parity risk and cold-start
  windows on every yield.

Recommendation: **A**. B only if A's provider changes are rejected, and only
behind the parity gate.

## Acceptance gates before any cutover

1. Parity: cosine >= 0.9999 for every item on a sample with multi-line and
   >2k-token texts, new path vs current `embed.preseen.ai` (both CPU and GPU paths
   of the new setup).
2. Yield test: with both GPUs at 2.5-2.6 GiB free, a ASR / small-model start
   succeeds after the embedding service yields, and embedding requests keep
   returning during the yield (served from CPU).
3. Non-sticky recovery: after a yield, a later burst (>= 8 texts) moves it back to
   GPU when VRAM is free.
4. Tests (unit, functional, integration) in `tests/`, README/CHANGELOG updated,
   committed by feature.

## Not decided

- Yield order among embedding / ASR / TTS (needs the ASR/TTS footprints measured on
  the same cards).
- Whether the provider change lands in `embedding-provider` (stardust) or the
  manager takes over the device policy entirely.
