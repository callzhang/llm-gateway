# Images: which instances take them

`VISION_GPU_IDS` (in `gateway.env`) says which GPUs' `qwen3.8-27b` instance takes images.

| Value | Meaning |
|---|---|
| unset | no restriction (previous behaviour) |
| `0` | only the instance on GPU 0; vLLM instances elsewhere start with `--language-model-only` |
| `none` | no images on this host |

| Host | Setting |
|---|---|
| gpu4 | `VISION_GPU_IDS=0` (the only vision instance in either region; used by `ai-services` `ocr_qwen`) |
| A100 | `VISION_GPU_IDS=none` |

## What a request gets

| Request | Answer |
|---|---|
| image, direct, host takes images | served by a vision GPU, never overflowed to the peer |
| image, direct, host takes none | 400 |
| image handed over by the peer, host takes none | 503 `gpu_busy` |
| image handed over by the peer, host takes images | 400: the peer's region takes none |
| image, vision GPU held by another model | 503 `gpu_busy` |

LiteLLM 1.99 moves on to the `order: 2` deployment after a 400 as well as after a 503 (measured 2026-11-26), which is why a handed-over image is refused: otherwise a CN image would be served by gpu4 anyway. The same fallback means a 400 from gpu4's engine for an image reaches the client as the A100's 503.

The A100 runs SGLang, whose `--language-only` loaded the same weights and KV as with vision (2026-11-26), so its engines are unchanged and images are only refused by model_manager. Refusing them keeps both image paths that ran SGLang out of memory (the processor and the vision encoder) from running.

Inline images over 4,194,304 pixels are also shrunk on the CPU before forwarding (`image_fit.py`).

## Deploying

1. **gpu4 first, then the A100.** An A100 that refuses images while gpu4 still has the old code gets its CN images served by gpu4 through the fallback.
2. On gpu4, restarting model_manager stops a running GPU 1 instance (it was started with vision) and the next request starts it text-only.
3. The A100 deployment of the same tree also renumbers its GPUs: see `gpu-guaranteed.md`.
4. Tell the callers: anything sending images to the CN entrance now gets 400.
