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
| image, direct, host takes images | served by a vision GPU, never queue-overflowed to the peer |
| image, direct, host takes none | 400, and LiteLLM tries the peer |
| image, direct, vision GPU held by another model | 503 `gpu_busy`, and LiteLLM tries the peer |
| web URL, direct, inline-only host | 400, and LiteLLM tries the peer |
| image handed over by the peer, host can take it | served by a vision GPU |
| image handed over by the peer, host takes none, or a web URL on an inline-only host | 503 `gpu_busy` |

An image goes to the other region only when the sending region cannot serve it (2026-11-28, Derek: image
requests should reach a multimodal instance, not fail). So a CN request with a web-URL image, which the A100
may not fetch, is processed on GPU4. LiteLLM moves on to the `order: 2` deployment after a 400 as well as
after a 503, and a handed-over request is never handed on again.

The A100 runs SGLang, whose `--language-only` loaded the same weights and KV as with vision (2026-11-26), so every A100 engine can take images and model_manager alone decides which instance gets them. Only card 1's does: SGLang has run out of GPU memory processing large images, and this keeps that to one instance.

An image URL must be `data:` or `http(s):`, whatever the vision setting (SGLang would open a path or `file://` URL from the server's disk); anything else is a 400.

Inline images over 4,194,304 pixels are also shrunk on the CPU before forwarding (`image_fit.py`).

context_fit adds each image's tokens to the prompt count (`image_fit.image_tokens`: about 400 for 0.4 Mpx, 4,096 at the 4 Mpx cap; an http URL counts as the largest).

`IMAGE_URLS_INLINE_ONLY=1` accepts only base64 `data:` URLs, and no video or audio parts: the engine then fetches nothing, and every image passes that shrink. The A100 sets it, because its engines could not reach public URLs anyway (`IPAddressDeny`, `region-cn.md`) while loopback and tailnet addresses were reachable.

## Deploying

1. On the A100 the settings are in `/etc/llm-model-manager-cn-27b-scaleout.env`; restart `llm-model-manager` (system unit) after a change.
2. On gpu4, restarting model_manager stops a running GPU 1 instance (it was started with vision) and the next request starts it text-only.
3. Changing which regions take images changes which region serves one (table above); tell the callers.
