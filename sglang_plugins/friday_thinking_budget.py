"""SGLang custom logit processor that caps qwen3.8-27b thinking at a per-request token budget.

Install next to SGLang (run_qwen38_27b_sglang.sh puts this directory on PYTHONPATH) and start the
server with --enable-custom-logit-processor. model_manager sends the processor's reference
(SGLANG_QWEN38_THINKING_BUDGET_PROCESSOR) with custom_params.thinking_budget whenever a request
to an SGLang backend carries vLLM's thinking_token_budget.

Why not SGLang's own Qwen3ThinkingBudgetLogitProcessor:
- it hardcodes Qwen3's <think>/</think> ids (151667/151668); qwen3.8 has them at 248068/248069, so
  it never sees the thinking section (measured 2026-11-16: budget 64 -> 538 reasoning tokens);
- it decides from req.output_ids, which under SGLang's overlap scheduler (the default) does not yet
  hold the token sampled one step earlier. It then forces </think> twice; the reasoning parser splits
  on the first and the second starts the answer ("</think>\\n\\n## Plan ...").
This processor keeps its own per-request state in the request's custom_params dict, which SGLang
creates once per request and passes again at every step.
"""
from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor

_NEWLINE_FORCED = "_friday_newline_forced"
_THINKING_CLOSED = "_friday_thinking_closed"


class Qwen38ThinkingBudgetLogitProcessor(CustomLogitProcessor):
    THINKING_START_TOKEN_ID = 248068
    THINKING_END_TOKEN_ID = 248069
    NEW_LINE_TOKEN_ID = 198

    def __call__(self, logits, custom_param_list):
        for i, params in enumerate(custom_param_list or []):
            if not params:
                continue
            budget = params.get("thinking_budget")
            if type(budget) is not int or budget < 0:
                continue
            req = params.get("__req__")
            if req is None:
                continue
            if params.get(_THINKING_CLOSED):
                # Thinking is over (we closed it); a second </think> would start the answer.
                logits[i, self.THINKING_END_TOKEN_ID] = -float("inf")
                continue
            if self.THINKING_END_TOKEN_ID in req.output_ids:
                params[_THINKING_CLOSED] = True  # the model closed thinking itself
                logits[i, self.THINKING_END_TOKEN_ID] = -float("inf")
                continue
            ids = [*req.origin_input_ids, *req.output_ids]
            if self.THINKING_START_TOKEN_ID not in ids:
                continue
            thought = len(ids) - ids.index(self.THINKING_START_TOKEN_ID) - 1
            if thought < budget:
                continue
            logits[i, :] = -float("inf")
            ends_with_newline = bool(req.output_ids) and req.output_ids[-1] == self.NEW_LINE_TOKEN_ID
            if not ends_with_newline and not params.get(_NEWLINE_FORCED):
                params[_NEWLINE_FORCED] = True
                logits[i, self.NEW_LINE_TOKEN_ID] = 0.0
            else:
                params[_THINKING_CLOSED] = True
                logits[i, self.THINKING_END_TOKEN_ID] = 0.0
        return logits
