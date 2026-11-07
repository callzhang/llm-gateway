# Changelog

## 2026-11-07

### Added
- Cross-region overflow in `model_manager`: with `OVERFLOW_QUEUE_DEPTH=N` (default 0 = off), new chat
  work for a model whose summed vLLM queue has reached N gets a `503 queue_overflow`, so LiteLLM's
  order-2 deployment (the other region's gateway) takes it. Sticky hits stay local for their prefix
  cache, and a request carrying `x-gateway-overflow` (set by the peer's deployment) is always served,
  so the two regions cannot bounce a request between them.

## 2026-11-03

### Added
- `scripts/litellm_mem_watch.sh` and the root `llm-litellm-mem-watch` timer: when LiteLLM's memory
  passes 2 GiB, capture py-spy stacks, process memory, :8900 connections, running Postgres queries and
  the log tail into `logs/mem-captures/`, so the next spike like 2026-11-01's 45.8 GB has evidence.

### Changed
- LiteLLM memory cap lowered from `MemoryHigh=6G`/`MemoryMax=8G` to 3G/4G; 2.5 days of samples
  peaked at 1.33 GB.

## 2026-11-01

### Fixed
- The LiteLLM proxy could not restart after the 01:35 OOM kill: `~/.cache/prisma-python` had been
  wiped, so the query engine path generated into `prisma/client.py` pointed at a missing file and every
  start exited 3. The Prisma client is now generated with `PRISMA_BINARY_CACHE_DIR` inside the conda env,
  and the `llm-litellm` unit sets the same directory. README documents regenerating after upgrades.
## 2026-10-22

### Changed
- Context-window fitting moved from the LiteLLM `trim_hook` pre-call hook to `model_manager`
  (`context_fit.py`), the last hop before vLLM. LiteLLM's Responses bridge appends the stored
  `previous_response_id` history after its pre-call hooks, so the hook could count that history but
  not trim it, and a long chain still hit vLLM's 400. `model_manager` receives the finished chat body,
  so the same policy now also covers `previous_response_id` history, Open WebUI and direct callers.
- The policy itself is unchanged (cap the output if that loses at most ~9%, else drop thinking, stub
  near-duplicates, cap tool arguments/results, then drop the oldest turns).
- The sticky-routing key is taken from the client's original messages before fitting.
- `run_litellm.sh` no longer exports `PYTHONPATH`: no remaining callback imports a sibling module.

### Added
- `scripts/context_fit_stats.py` and the `llm-context-fit-stats` systemd timer: a daily report in
  `logs/context_fit_stats.report` of how many chat requests context_fit had to shrink, by furthest stage.

### Removed
- `trim_hook.py`, its `callbacks` entry in `config.yaml`, and its Responses-shape handling. A native
  `/v1/responses` call sent directly to `model_manager` (not via LiteLLM) is no longer fitted.

### Measured
- Token counting costs ~1.5-2 µs/token; a 169k-token request that needs shrinking takes ~2 s (11
  counts), while the event loop lags at most 13 ms. See the README for details.

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
