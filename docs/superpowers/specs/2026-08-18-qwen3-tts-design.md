# Qwen3-TTS Gateway Design

## Goal

Add `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` to the existing GPU4 LLM gateway as an OpenAI-compatible text-to-speech model. The first release must serve both Open WebUI and authenticated external API clients without weakening the existing chat-model GPU isolation.

## Scope

The first release exposes the model as `qwen3-tts-1.7b-customvoice` through:

```text
POST https://llm-api.preseen.ai/v1/audio/speech
```

Supported request fields are `model`, `input`, `voice`, `instructions`, and `response_format`. Preset CustomVoice speakers and instruction-based speaking style are included. Voice cloning, uploaded reference audio, VoiceDesign, and Base checkpoints are excluded because the installed LiteLLM speech route does not reliably preserve Qwen3-TTS extension fields.

## Architecture

The request path remains:

```text
Client or Open WebUI
  -> Cloudflare tunnel
  -> LiteLLM /v1/audio/speech
  -> model_manager
  -> vLLM-Omni Qwen3-TTS backend
  -> streamed or complete audio response
```

LiteLLM remains the public authentication, model authorization, quota, and audit boundary. The built-in speech endpoint is used so existing LiteLLM virtual keys apply without creating a second credential system.

Qwen3-TTS runs in a dedicated `.venv-tts`. The existing gateway `.venv` and its vLLM installation are not upgraded or replaced. The startup script must honor `VLLM_CUDA_DEVICE`, `VLLM_PORT`, and any measured memory controls supplied by `model_manager`.

## GPU Lifecycle

Qwen3-TTS participates in the existing single-GPU slot allocator. A slot may run either one chat backend or one Qwen3-TTS backend, never both. This preserves mutual exclusion even though the TTS model is smaller than the chat models.

The first request may cold-start the backend and wait for readiness. While an audio response is being generated or streamed, the request remains active and prevents idle reclamation. After the configured idle timeout, the backend is unloaded and the slot becomes available to chat models.

Memory admission values must come from a measured GPU4 run. The implementation must not infer a production limit solely from parameter count. A failed spawn must release its slot and return a structured 503 response.

## API Contract

The minimum accepted request is:

```json
{
  "model": "qwen3-tts-1.7b-customvoice",
  "input": "你好，这是语音合成测试。",
  "voice": "Vivian",
  "response_format": "wav"
}
```

`instructions` is optional. The gateway must retain the upstream audio content type and stream or copy the binary body without JSON transformation. A configurable input-length limit protects the public endpoint from unbounded synthesis requests. Invalid models and oversized inputs return OpenAI-style JSON errors before backend scheduling. Voice and format validation remains authoritative in vLLM-Omni, and its structured error is forwarded through the gateway.

## Failure Handling and Observability

- No free compatible GPU: return `503 gpu_busy` without queueing indefinitely.
- Backend startup failure: release the claimed slot and return `503 startup_failed`.
- Upstream generation failure: preserve the existing backend recycle policy where applicable and return a gateway error without emitting a partial file as a successful response.
- Client disconnect: stop forwarding and ensure the active-request count is eventually decremented.
- Logs record model, route, response status, elapsed time, selected slot, and output byte count. They must not record synthesized text, API keys, or audio content.

## Security

- External access continues through `llm-api.preseen.ai`; no new hostname or public listener is introduced.
- Existing LiteLLM virtual keys are required.
- The vLLM-Omni backend binds to loopback and is not exposed directly.
- Request text and generated audio are transient and are not persisted by the gateway.
- The implementation does not add voice cloning, which avoids accepting biometric reference audio in the first release.

## Testing

Implementation follows test-driven development.

Automated tests must first fail and then pass for:

1. Qwen3-TTS model registration and startup-script selection.
2. `/v1/audio/speech` model extraction and routing.
3. Binary response content type and body forwarding.
4. Input-length rejection before backend startup.
5. Active audio requests preventing idle unload.
6. Slot release after startup or generation failure.

GPU4 acceptance requires fresh evidence for:

1. A LiteLLM virtual key can call the public `/v1/audio/speech` endpoint.
2. The response is a non-empty RIFF/WAV file with a readable sample rate and duration.
3. The waveform contains non-silent audio.
4. GPU memory is allocated to the intended card during synthesis and released after unload.
5. A chat model can wake and serve a completion after TTS releases the slot.
6. Open WebUI can use the same endpoint and credentials.

## Deployment and Rollback

GPU4 currently contains two commits beyond the local checkout. They must be fetched and preserved before implementation; deployment must never overwrite or discard them.

Deployment is incremental:

1. Install the isolated TTS environment and model dependencies.
2. Run a loopback-only backend smoke test.
3. Deploy model-manager and LiteLLM configuration changes.
4. Restart only the required user services and verify process recovery.
5. Run internal, authenticated public, GPU lifecycle, and chat-regression checks.

Rollback removes the TTS model from LiteLLM and `model_manager`, restores the prior tracked service configuration, and restarts the affected services. Downloaded model weights and the isolated environment may remain as inert cached data unless removal is explicitly requested.

## Completion Criteria

The work is complete only when automated tests pass, the public authenticated API produces validated audible WAV output on GPU4, Open WebUI succeeds with the same backend, GPU reclamation is observed, and a post-TTS chat request succeeds. Any missing external, audio-quality, GPU-release, or chat-regression evidence must be reported as incomplete rather than inferred.
