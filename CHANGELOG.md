# Changelog

## 2026-10-06

### Fixed
- `/v1/responses` requests whose prompt plus requested output exceeded the model's context window
  (e.g. 98,305 + 32,768 > 131,072) were rejected by vLLM with a 400 that LiteLLM passed through,
  and the client retried the same request 216 times. `trim_hook` now fits both `/v1/chat/completions`
  and `/v1/responses` requests before they reach vLLM.
- Invalid-JSON tool-call arguments no longer crash token counting with a 500.

### Changed
- `trim_hook` counts input exactly with the model's tokenizer and chat template (tools and
  `previous_response_id` history included) instead of an ASCII/CJK estimate.
- When the output does not fit: shrink the output cap if that loses at most ~9%; otherwise shrink the
  input in stages — drop old thinking, stub >= 98% similar messages (keeping the latest copy), cap tool
  arguments/results to `tool_cap_tokens`, then drop the oldest turns. Requests that fit are untouched.
- `config.yaml`: `model_info.max_context_window_tokens` is replaced by `max_input_tokens`,
  `max_output_tokens`, `tokenizer`, `tokenizer_revision` and optional `tool_cap_tokens`; LiteLLM registers
  the first two into `litellm.model_cost`. A model that declares a window without a tokenizer fails at
  startup.
- Callback order: `previous_response_history_hook` now runs before `trim_hook`.
- `run_litellm.sh` exports its directory on `PYTHONPATH` (callbacks are loaded by file path).

### Known gaps
- Image tokens expand inside vLLM; only their placeholder is counted (512-token margin).
- History behind `previous_response_id` is counted but cannot be trimmed.
- LiteLLM 1.99 leaves `incomplete_details` empty on a capped Responses answer (fixed upstream in 1.104).
