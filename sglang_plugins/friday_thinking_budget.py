"""SGLang custom logit processor that caps qwen3.8-27b thinking at a per-request token budget.

Install next to SGLang (the A100 copies it into the SGLang venv's site-packages) and start the
server with --enable-custom-logit-processor. model_manager sends the processor's reference
(SGLANG_QWEN38_THINKING_BUDGET_PROCESSOR) with custom_params.thinking_budget whenever a request
to an SGLang backend carries vLLM's thinking_token_budget.

SGLang's own Qwen3ThinkingBudgetLogitProcessor hardcodes Qwen3's <think>/</think> ids
(151667/151668). Qwen3.8's tokenizer has them at 248068/248069, so that class never sees the
thinking section and never caps it (measured 2026-11-16: budget 64 -> 538 reasoning tokens;
with these ids -> 65-67).
"""
from sglang.srt.sampling.custom_logit_processor import ThinkingBudgetLogitProcessor


class Qwen38ThinkingBudgetLogitProcessor(ThinkingBudgetLogitProcessor):
    THINKING_START_TOKEN_ID = 248068
    THINKING_END_TOKEN_ID = 248069
    NEW_LINE_TOKEN_ID = 198
