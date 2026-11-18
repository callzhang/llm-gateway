# GPU neighbour reserve

`GPU_NEIGHBOUR_RESERVE_MIB` keeps part of one GPU out of every vLLM spawn's plan, so that a neighbour process with a fixed footprint can always start there. The neighbour today is the embedding-provider (`embed.preseen.ai`, stardust's `embedding-provider.service`).

```
GPU_NEIGHBOUR_RESERVE_MIB=1=4096      # gateway.env; "<gpu>=<MiB>", several joined by ";"
```

Unset, nothing changes: chat models take whatever is free at spawn, up to `CHAT_GPU_MEM_UTIL_CEILING` (0.95).

## Why

The embedding-provider serves from CPU and moves bulk batches to a GPU worker. On 2026-11-17 both 5090s ran `qwen3.8-27b` at util 0.950, which leaves about 0.9 GiB per card. Every scale-up then failed with CUDA OOM, the provider stayed on CPU, and one CPU queue of long documents made recall embeddings take 60–130 s. The 0.95 ceiling (8b8072f, 2026-10-03) assumed neighbours could live on CPU; for the embedder under document ingest they cannot.

## What it does

For a GPU with a reserve, a vLLM process may plan at most

```
room = total − reserve − (vLLM memory already on the GPU) − VLLM_PROCESS_OVERHEAD_MIB (762)
```

and `_gpu_mem_util_for_spawn` caps util at `floor(room / total, 0.001)`. The reserve is counted against the card's total, not its free memory, so it holds whether or not the neighbour is on the GPU, and a neighbour already using its slice is not counted twice. Spawns are refused rather than clamped up into the reserve: if the model's min-viable floor exceeds `room`, the spawn fails with "does not fit beside the … neighbour reserve". Co-resident models (ASR) skip a GPU whose room is reserved. GPUs without a reserve are unaffected.

On a 32,607 MiB RTX 5090 with `1=4096`, the 27B on GPU 1 gets util 0.851 instead of 0.950. Its footprint is 0.851 × 32,607 + 762 = 28,511 MiB, leaving 4,096 MiB free. That is about 3.2 GiB less KV cache on that replica only. The 27B floor is 24.83 GiB, so a reserve of up to 6,419 MiB still lets it start there; for the 35B heretic (floor 26.74 GiB) the limit is 4,463 MiB.

## Sizing

Size the reserve from the neighbour's measured peak footprint, not from a guess, and change it together with the setting that bounds that peak. For the embedding-provider that setting is `CUDA_MAX_FORWARD_TOKENS`. The footprint grows about 1.3 GiB per 8192 tokens per forward. Measured on an RTX 5090 with jina-v5-small at `MAX_LENGTH=8192` (embedding-provider `docs/benchmarks/2026-11-17-interactive-queue-and-cuda-peak.md`):

| tokens per forward | process footprint |
|---:|---:|
| loaded only | 1,966 MiB |
| 8,192 | 3,438 MiB |
| 16,384 | 4,848 MiB |
| 32,768 | 8,048 MiB |
| 65,536 | 14,448 MiB |

With `CUDA_MAX_FORWARD_TOKENS=8192`, the reserve of 4,096 MiB is the 3,438 MiB peak plus 658 MiB of headroom for allocator fragmentation. The embedder must be pinned to the reserved card (`CUDA_VISIBLE_DEVICES=1` in its env); otherwise it picks the card with the most free VRAM.

## Verify

After a spawn on the reserved GPU, `nvidia-smi --query-gpu=index,memory.free --format=csv` should show at least the reserve free on that card while the embedder is on CPU. With the embedder scaled up (`curl -s 127.0.0.1:7997/healthz` → `runtime.loaded_device == "cuda"`, `cuda_device_index == "1"`), the card's free memory should stay above zero. The model_manager log line for the spawn shows the capped util.
