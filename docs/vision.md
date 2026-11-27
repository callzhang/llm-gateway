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
| A100 | `VISION_GPU_IDS=1`, `IMAGE_URLS_INLINE_ONLY=1` (images only as base64 `data:` URLs) |

## What a request gets

| Request | Answer |
|---|---|
| image, direct, host takes images | served by a vision GPU, never overflowed to the peer |
| image, direct, host takes none | 400 |
| image handed over by the peer, host takes none | 503 `gpu_busy` |
| inline image handed over by the peer, host takes images | 503 `gpu_busy`: the peer lacked capacity |
| other image handed over by the peer, host takes images | 400: the peer refused its URL |
| image, vision GPU held by another model | 503 `gpu_busy` |

LiteLLM 1.99 moves on to the `order: 2` deployment after a 400 as well as after a 503 (measured 2026-11-26), which is why a handed-over image is never served: otherwise one region's image would run in the other. The client sees the second region's answer.

The A100 runs SGLang, whose `--language-only` loaded the same weights and KV as with vision (2026-11-26), so every A100 engine can take images and model_manager alone decides which instance gets them. Only card 1's does: SGLang has run out of GPU memory processing large images, and this keeps that to one instance.

An image URL must be `data:` or `http(s):`, whatever the vision setting (SGLang would open a path or `file://` URL from the server's disk); anything else is a 400.

Inline images over 4,194,304 pixels are also shrunk on the CPU before forwarding (`image_fit.py`).

`IMAGE_URLS_INLINE_ONLY=1` accepts only base64 `data:` URLs: the engine then fetches nothing, and every image passes that shrink. The A100 sets it, because its engines could not reach public URLs anyway (`IPAddressDeny`, `region-cn.md`) while loopback and tailnet addresses were reachable.

## Deploying

1. On the A100 the settings are in `/etc/llm-model-manager-cn-27b-scaleout.env`; restart `llm-model-manager` (system unit) after a change.
2. On gpu4, restarting model_manager stops a running GPU 1 instance (it was started with vision) and the next request starts it text-only.
3. Changing which regions take images changes what a handed-over image gets (table above); tell the callers.
