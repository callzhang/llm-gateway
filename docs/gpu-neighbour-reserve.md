# GPU neighbour reserve

`GPU_NEIGHBOUR_RESERVE_MIB` keeps part of one GPU out of every vLLM spawn's plan, so that a neighbour process with a fixed footprint can always start there. The neighbour today is the embedding-provider (`embed.preseen.ai`, stardust's `embedding-provider.service`).

```
GPU_NEIGHBOUR_RESERVE_MIB=1=3900      # gateway.env; "<gpu>=<MiB>", several joined by ";"
```

gpu4 runs `1=3900` since 2026-11-20 (Derek): the largest round value under the 35B heretic's limit on GPU 1 (about 3,935 MiB), 336 MiB above the embedder's live peak of 3,564 MiB. With it the 27B on GPU 1 gets util 0.842 (0.842 × 32,149 + 1,176 = 28,246 MiB, 3,903 MiB free). The numbers below for `1=4096` show the formula.

Unset, nothing changes: chat models take whatever is free at spawn, up to `CHAT_GPU_MEM_UTIL_CEILING` (0.95).

## Why

The embedding-provider serves from CPU and moves bulk batches to a GPU worker. On 2026-11-17 both 5090s ran `qwen3.8-27b` at util 0.950, which leaves about 0.9 GiB per card. Every scale-up then failed with CUDA OOM, the provider stayed on CPU, and one CPU queue of long documents made recall embeddings take 60–130 s. The 0.95 ceiling (8b8072f, 2026-10-03) assumed neighbours could live on CPU; for the embedder under document ingest they cannot.

## What it does

For a GPU with a reserve, a vLLM process may plan at most

```
cuda_total = memory.total − memory.reserved           # what vLLM multiplies util by
room       = cuda_total − reserve − (vLLM memory already on the GPU) − VLLM_PROCESS_OVERHEAD_MIB (1,176)
```

and `_gpu_mem_util_for_spawn` caps util at `floor(room / cuda_total, 0.001)`. `memory.reserved` is the memory the NVIDIA driver and firmware hold (458 MiB on gpu4's GPU 1, 491 MiB on GPU 0); no process can allocate it, CUDA leaves it out of the card's total, and vLLM's startup log shows that total ("Free memory on device (30.9/31.4 GiB)"). If it cannot be read, a spawn on the reserved GPU is refused. The reserve is counted against the card's total, not its free memory, so it holds whether or not the neighbour is on the GPU, and a neighbour already using its slice is not counted twice. Spawns are refused rather than clamped up into the reserve: if the model's min-viable floor exceeds `room`, the spawn fails with "does not fit beside the … neighbour reserve". Co-resident models (ASR) skip a GPU whose room is reserved. GPUs without a reserve are unaffected.

`VLLM_PROCESS_OVERHEAD_MIB` is the vLLM process's nvidia-smi footprint beyond util × cuda_total, measured on the 27B after warm-up: 1,176 MiB at 0.95 on GPU 0, 1,142 at 0.95 on GPU 1, 1,165 at 0.851 on GPU 1 (2026-11-17 to 11-20). The largest is used.

Until 2026-11-20 the formula used `memory.total` and an overhead of 762 MiB measured against it, which left the driver's memory inside the slice: with `1=4096` the 27B started at util 0.851 and, once warm, left 3,657 MiB free on GPU 1, 93 MiB above the embedder's 3,564 MiB peak.

On GPU 1 (32,607 MiB, 458 reserved, cuda_total 32,149) with `1=4096`, the 27B gets util 0.836 instead of 0.950. Its footprint is 0.836 × 32,149 + 1,176 = 28,053 MiB, leaving 4,096 MiB free, and its KV cache is about 7.1 GiB instead of 10.6 GiB (about 196k tokens instead of 293k; 1.5 instead of 2.2 requests of 131,072 tokens at once). Its context length per request is unchanged.

The largest reserve a model can start beside is `cuda_total − 1,176 − min_viable × cuda_total`, where `min_viable` is the model's floor in GiB divided by `memory.total`, rounded up to 0.001, and rounding of the capped util costs up to 0.001 × cuda_total more. On GPU 1: the 27B (floor 24.83 GiB, util 0.780) can start beside up to about 5,860 MiB; the 35B heretic (floor 26.74 GiB, util 0.840) up to about 3,935 MiB, so with `1=4096` it no longer starts on GPU 1.

## Sizing

Size the reserve from the neighbour's measured peak footprint, not from a guess, and change it together with the setting that bounds that peak. For the embedding-provider that setting is `CUDA_MAX_FORWARD_TOKENS`. The footprint grows about 1.3 GiB per 8192 tokens per forward. Measured on an RTX 5090 with jina-v5-small at `MAX_LENGTH=8192` (embedding-provider `docs/benchmarks/2026-11-17-interactive-queue-and-cuda-peak.md`):

| tokens per forward | process footprint |
|---:|---:|
| loaded only | 1,966 MiB |
| 8,192 | 3,438 MiB |
| 16,384 | 4,848 MiB |
| 32,768 | 8,048 MiB |
| 65,536 | 14,448 MiB |

With `CUDA_MAX_FORWARD_TOKENS=8192`, the reserve of 4,096 MiB is the 3,438 MiB peak plus 658 MiB of headroom for allocator fragmentation. The live worker peaked at 3,564 MiB on 2026-11-18 (8 texts of 8,192 tokens, one per forward). The embedder must be pinned to the reserved card (`CUDA_VISIBLE_DEVICES=1` in its env); otherwise it picks the card with the most free VRAM.

## Verify

After a spawn on the reserved GPU, `nvidia-smi --query-gpu=index,memory.free --format=csv` should show at least the reserve free on that card while the embedder is on CPU. With the embedder scaled up (`curl -s 127.0.0.1:7997/healthz` → `runtime.loaded_device == "cuda"`, `cuda_device_index == "1"`), the card's free memory should stay above zero. The model_manager log line for the spawn shows the capped util.
