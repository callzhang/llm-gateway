# Changelog

## Unreleased

### Security
- An image URL must be `data:` or `http(s):`; a path or `file://` URL is a 400.
- A100: SGLang's internal sockets bind loopback, the gateway unit takes only host and tailnet connections, and
  `/data/derek` is no longer world-writable (`docs/region-cn.md`).

### Added
- A100: `qwen3.8-27b` takes images again, on card 1's instance, as base64 `data:` URLs only
  (`IMAGE_URLS_INLINE_ONLY=1`). A handed-over inline image answers 503 on a host that takes images (`docs/vision.md`).
- `qwen3.8-27b`: model_manager shrinks inline (data: URL) images over 4,194,304 pixels on the CPU before forwarding
  (`image_fit.py`). The engine caps images there anyway but resized them on the GPU, where very large sources (up to
  128 megapixels on the CN key) ran the A100's SGLang out of memory and killed the instance. Images by http URL pass
  through unchanged; one Pillow refuses to decode (over ~179 megapixels) is a 400.
- `VISION_GPU_IDS`: only the listed GPUs' `qwen3.8-27b` instance takes images (`none`: the host takes none). Image
  requests go to those instances and never overflow to the other region; see `docs/vision.md`.
- `GUARANTEED_GPU_UUIDS`: cards cleared of foreign processes before a primary model starts there; see
  `docs/gpu-guaranteed.md`.

### Changed
- Engines start with `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so a GPU id means the nvidia-smi index. On the A100 this needs
  the pins renumbered in the same deployment (`docs/gpu-guaranteed.md`).

### Security
- Chat models accept only `POST /v1/chat/completions`, `/v1/completions` and `/v1/responses`, and a client body may
  not set `custom_logit_processor`, `mm_processor_kwargs` or `chat_template`.
- The engines accept only `ENGINE_API_KEY` (new, required in `gateway.env`; model_manager and the launch scripts
  refuse to start without it), on every route but `/health` and `/metrics`. model_manager replaces the caller's
  `Authorization` with it. It replaces the fixed key in the launch scripts and `ASR_GPU_API_KEY`. A restart stops
  engines started with anything else instead of adopting them.

### Fixed
- `retry_breadcrumb_hook`: spend logs no longer hold other callers' requests. LiteLLM 1.99 kept retry breadcrumbs,
  messages and images included, on one list per Router, and the overflow fallback wrote them even with
  `num_retries: 0`. Rows already stored are unchanged (`docs/spend-log-retry-leak.md`).

## 2026-11-22

### Fixed
- `previous_response_hook` answers before the first SSE keepalive, so its 410 stays a real HTTP status. It waited up
  to 30 s for history inside LiteLLM's keepalive wrapper; with pings at 10 s, US prod Memory's streaming
  continuations got a 200, pings and a generic stream error, and its extraction jobs failed for good. An id this
  proxy never issued (it must decrypt with this proxy's key to a managed id naming one of its deployments) is now
  410 at once without a lookup; its own ids wait at most 12 s (`PREVIOUS_RESPONSE_WAIT_SECONDS`), the keepalive
  starts at 15 s, and `proxy_batch_write_at: 2` writes spend-log rows every 2-7 s instead of 10-15 s so the history
  lands inside the wait.

### Known
- A streamed Responses answer cannot be continued with `previous_response_id` on LiteLLM 1.99's `custom_openai`
  bridge: its final id points at the stream's placeholder `resp_<uuid>`, which no spend-log row records, so the
  continuation gets 410. Non-streamed answers continue normally. Streaming clients must resend the conversation.

### Added
- Priority 0 at model_manager's gate is now a grant held by a LiteLLM virtual key. `request_priority_hook.py`
  runs a key whose metadata has `gateway_priority: P` at P when the request sends none and never better than P;
  any other key's 0 becomes 1. model_manager raises the peer region's requests (which skip GPU4's LiteLLM) to
  `PEER_PRIORITY_FLOOR` = 1. Before this, any client could claim 0: on 2026-11-22, 76 requests from the A100 and
  122 on the shared `memory-ci-controller` key did. New key `friday-memory-us-prod` (US production friday-memory)
  holds `gateway_priority: 0`; it is stored on GPU4 in `~/.config/llm-gateway/keys/`.
- Cross-region overflow counts only the requests queued ahead of the incoming one: gate waiters with a larger
  priority number are left out of the queue compared with `OVERFLOW_QUEUE_DEPTH`. The A100's requests are never
  turned away here, so while they queued on GPU4 every local request, a prod priority-0 one included, was sent
  across at once instead of jumping the queue. A normal (100) request also stays local while only evals (200) wait.
  The depth itself is unchanged.
- Streaming requests no longer 524 while queued. Cloudflare gives up after ~100 s without response bytes, and
  LiteLLM wrote nothing until model_manager's first chunk, so a streaming request held at the gate for minutes
  failed. `sse_keepalive_ping_interval_seconds: 15` turns on LiteLLM 1.99's TTFT keepalive: after 15 s of silence
  the client gets 200 SSE headers and a `: ping` comment every 15 s (also in gaps mid-stream). A failure found after
  the first ping arrives as a `data: {"error": ...}` frame under the 200. Non-streaming requests are unchanged.

### Fixed
- `OVERFLOW_QUEUE_DEPTH` is now per running instance: model_manager sends new chat work to the peer region once the
  model's summed queue reaches the depth times its instances, not the depth itself. On the A100, after scale-out to
  four 27B replicas (cards 0/1/2/6), the flat 4 still sent ~40 requests a minute to GPU4 while each replica ran
  10-12 of 16 and queued about one. GPU4 (two instances) now overflows at 8 instead of 4 once restarted.

## 2026-11-20

### Added
- `GPU_NEIGHBOUR_RESERVE_EXEMPT_MODELS` (comma-separated, unset by default) lets listed models ignore
  `GPU_NEIGHBOUR_RESERVE_MIB`. gpu4 lists the 35B heretic: held to GPU 1's 3,900 MiB slice it would start at about
  util 0.84 (169k KV tokens) instead of 0.95 (454k). While it holds GPU 1 the embedder stays on CPU.

### Fixed
- `GPU_NEIGHBOUR_RESERVE_MIB` now leaves the full reserve free in nvidia-smi. vLLM multiplies util by CUDA's total,
  which is `memory.total` minus the memory the driver and firmware reserve (458 MiB on gpu4's GPU 1), but the room
  was computed against `memory.total` with an overhead calibrated the same way. With `1=4096` the 27B started at
  0.851 and, once warm, left 3,657 MiB free on GPU 1, 93 MiB above the embedder's 3,564 MiB peak. The room is now
  `memory.total − memory.reserved − reserve − other vLLM − 1,176 MiB` (overhead re-measured against CUDA's total,
  the largest of three 27B readings), so the 27B gets 0.836. A spawn on a reserved GPU is refused if
  `memory.reserved` cannot be read. With `1=4096` the 35B heretic (floor 0.840) no longer fits on GPU 1; see
  docs/gpu-neighbour-reserve.md for each model's largest reserve.

## 2026-11-17

### Added
- `GPU_NEIGHBOUR_RESERVE_MIB` ("<gpu>=<MiB>;…", unset by default) keeps that much of a GPU out of every vLLM
  spawn's plan, counted against the card's total so it holds whether or not the neighbour is on the GPU. With both
  5090s at util 0.950 (about 0.9 GiB left each) the embedding-provider's CUDA scale-up failed with OOM on every try
  on 2026-11-17, and long documents on its CPU queue made recall embeddings take 60-130 s. A spawn that cannot fit
  its min-viable floor beside the reserve is refused rather than clamped into it; co-resident models skip a reserved
  GPU without room. Sizing (from the embedder's measured footprint at `CUDA_MAX_FORWARD_TOKENS=8192`: 3,438 MiB) and
  the KV cost (`1=4096` caps the 27B on GPU 1 at util 0.851, about 3.2 GiB less KV there):
  `docs/gpu-neighbour-reserve.md`.

### Fixed
- LiteLLM no longer truncates the prompts and responses it stores (`MAX_STRING_LENGTH_PROMPT_IN_DB` set to
  10,000,000 in `run_litellm.sh`). It rebuilds a `previous_response_id` conversation from those rows, and the
  default 2,048-character cut sent continued requests with gutted instructions and earlier answers: on the A100,
  CN memory's continued extractions thought 10-20k tokens and returned no answer or failed in 87 of 142 calls
  (2026-11-17 00:00-07:00Z). Rows stored before the change stay truncated.
- SGLang JSON-schema output no longer allows unlimited whitespace (`--constrained-json-disable-any-whitespace`).
  A string value closed early by a bare `"` left the model only whitespace to emit, and CN memory's judges ran
  to their 16,000-token ceiling (about 575 s each, up to 25 an hour).

## 2026-11-16

### Fixed
- The A100's SGLang backend for qwen3.8-27b now honours a thinking budget. SGLang ignores vLLM's
  `thinking_token_budget`, so since the A100 switched the 27B to SGLang (2026-11-14 08:08Z) CN's memory
  jobs thought without limit: community summaries returned exactly 8,192 output tokens (5,918-8,192 of
  them reasoning) or were cut by the client at 300 s. model_manager now turns `thinking_token_budget`
  into SGLang's thinking-budget processor for an `engine: sglang` model. The processor is our own
  `sglang_plugins/friday_thinking_budget.py`: SGLang's `Qwen3ThinkingBudgetLogitProcessor` hardcodes
  Qwen3's think-token ids (151667/151668), and qwen3.8's are 248068/248069, so it never applied
  (measured on the A100: budget 64 -> 538 reasoning tokens with SGLang's class). It also reads
  `req.output_ids`, which under SGLang's overlap scheduler lags one step, so it forced `</think>` twice
  and the second started the answer; ours keeps its own state in the request's `custom_params`.
  Measured on an A100 test instance started from `run_qwen38_27b_sglang.sh`: budgets 64/256/300 give
  65/257/301 reasoning tokens, no `</think>` in any answer, with and without a JSON schema.

### Added
- The A100's SGLang engine support, which had run there uncommitted since 2026-11-14:
  `ModelConfig.engine`, SGLang process detection and queue metric in model_manager,
  `QWEN27B_ENGINE=sglang`, and `run_qwen38_27b_sglang.sh` (now with `--enable-custom-logit-processor`).

## 2026-11-08

### Fixed
- Sticky routing no longer pins a conversation to a slot that cannot serve it. A sticky binding
  lapses after `STICKY_IDLE_TTL` (default 300 s) without a request, and a sticky hit whose own slot
  has requests queued at the priority gate is turned away with `503 queue_overflow` when the model's
  queue is at `OVERFLOW_QUEUE_DEPTH`, like any other request. Found on 2026-11-08: a follow-up with
  `previous_response_id` (110 chars, below the 2096-token prefix-cache block, so nothing to reuse)
  stayed on slot 0 behind the gate for 167-264 s while the A100 was idle. Requests that already
  crossed are still always served.
- The priority gate no longer holds a request past what Cloudflare allows. The gate waited without a limit and
  sent the client nothing meanwhile, so a request waiting over 125 s got a 524 that LiteLLM cannot hand over.
  Between 03:22 and 03:26 on 2026-11-08, slot 0 held 7 requests for 108-264 s while long requests handed over by the
  A100 region filled it. A local request now waits at most `GATE_MAX_WAIT_SECONDS` (default 90) and then gets 503
  `queue_overflow`, so LiteLLM tries the peer region. It bounds sticky requests too, even when
  the queue check above lets them through. It applies only when `OVERFLOW_QUEUE_DEPTH` is set, and never to work the peer region
  already handed over.
- Scale-out now also fires when every replica is busy, not only when vLLM queues. On 2026-11-07
  qwen3.8-27b ran 4-5 long-context agent requests on slot 1 for half an hour (vLLM waiting mostly 0,
  ~300 tok/s shared) while slot 0 was unloaded and GPU 0 empty; the queue tiers never fired, and
  fallback callers hit the 600s deadline and Cloudflare 524s. Every replica with at least
  `SCALE_OUT_BUSY_REQUESTS` (3) in flight now accrues toward scale-out over
  `SCALE_OUT_BUSY_SUSTAIN` (120s). A busy-only firing takes a free slot and never evicts another
  model; queue backlog keeps its eviction behaviour.

## 2026-11-07

### Changed
- Scale-out is no longer capped at two replicas. A model under sustained backlog adds one replica
  per firing while a free slot with room exists, so the 7-GPU A100 region can use more than two GPUs
  for one model. GPU4 has two slots, so its behaviour is unchanged.

### Fixed
- A model's minimum viable VRAM is now in GiB (`MODEL_MIN_VIABLE_GIB`, replacing the fraction-of-card
  `MODEL_MIN_GPU_MEM_UTIL`; the `GPU_MEM_UTIL_FLOOR` env fallback is now `MIN_VIABLE_GIB_FLOOR`) and is
  turned into a utilization from each GPU's total. The fractions were calibrated on the 32 GB 5090, so
  on a 40 GB A100 the 27B asked for 31.2 GiB and a card with 29.5 GiB free was refused; it now needs
  24.83 GiB (util 0.621). The spawn util, co-resident admission and reclaim all use it. On GPU4's cards
  the thresholds are unchanged (0.84 / 0.78 / 0.095).
- The CN region's vLLM (`/data/derek/venv-vllm029`) still had the stock Qwen tool parser. For a Responses
  request with tools, `tool_choice: auto` and a JSON-schema `text.format`, the stock parser adds no tool
  grammar (it needs `VLLM_ENFORCE_STRICT_TOOL_CALLING`, set in neither region), so the schema alone
  constrains the output and the model cannot call a tool: told to call `read_source`, A100 answered JSON
  3/3 times while GPU4 (patched 2026-10-25) called it 3/3. `scripts/install_qwen_parser.sh` installs `patches/qwen3_engine_tool_parser.py`
  into a given `VLLM_BIN`, after its tests pass under that interpreter, and it is installed on the A100.
- `test_qwen_native_output_grammar` moved to `tests_vllm/`: it tests a vLLM 0.29 patch and errored
  under the gateway env's vLLM 0.25.1, so `tests/` is now green in the gateway env.
- A second scale-out could start while the first replica was still loading (the guard counted only
  ready replicas). Scale-out and its accrual now wait for in-flight spawns.
- `/admin/kill|start|switch` answer 403 to non-loopback clients; :8002 is reachable over Tailscale
  and the A100 peer is a shared host.

### Added
- Cross-region overflow in `model_manager`: with `OVERFLOW_QUEUE_DEPTH=N` (default 0 = off), new chat
  work for a model whose summed vLLM queue has reached N gets a `503 queue_overflow`, so LiteLLM's
  order-2 deployment (the other region's gateway) takes it. Sticky hits stay local for their prefix
  cache, and a request carrying `x-gateway-overflow` (set by the peer's deployment) is always served,
  so the two regions cannot bounce a request between them.
- TTS and ASR are served in the CN region too, and each region lists the other as their order-2
  backup in LiteLLM. TTS fails over when no card is free (`503 gpu_busy`). ASR uploads of
  `ASR_OVERFLOW_MIN_SECONDS` (300) or more that would run on the CPU worker get `503 queue_overflow`
  when cross-region overflow is on. Because LiteLLM 1.99 drops `extra_headers` on audio calls, a
  request from another host now also counts as already crossed. `run_qwen3_tts_1_7b_customvoice.sh`
  finds its deploy config next to itself and takes `MODEL_CUDA_HOME`; `run_qwen3_asr_0_6b.sh`
  takes `ASR_VLLM_DISABLED_KERNELS` (Ampere has no FP8; vLLM 0.29 still picks a CUTLASS FP8
  kernel); `QWEN3_TTS_MIN_FREE_GIB` overrides the TTS VRAM gate (31.2 GiB measured on a 40 GB A100).
- Direct CN entrance on the A100: Caddy terminates TLS on :18900 with an optional source-IP allowlist (open to all sources)
  (`systemd/region-cn/Caddyfile`, `llm-tls-proxy.service`), and `LITELLM_HOST` lets the A100 bind
  LiteLLM to loopback. `scripts/region_cn_tls.sh` creates the DNS-only record and pushes the
  certificate issued on GPU4. Going through Cloudflare was measured slower for CN callers.

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
