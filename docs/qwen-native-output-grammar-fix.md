# Qwen reasoning and native JSON output (2026-10-24)

The deployed custom vLLM 0.29 parser built `Or(reasoning + required_tool, final_json)`.
With reasoning enabled, a valid `reasoning</think>\n\n{...}` could not be accepted:
choosing the reasoning prefix committed to a required tool call. This explains a
mechanism for repeated reads even after facts were sufficient. Native schema
visibility diagnostics alone were not reliable and are not the release fix.

The candidate uses the existing xgrammar Qwen reasoning prefix followed by
`Or(required_tool, final_json)`. No model route, tool authority, schema or retry
changes. Without reasoning, the original tool/JSON choice remains.

CPU regression uses the actual parser method and xgrammar matcher. Before fix,
JSON after reasoning failed, tools and strict negatives passed. After fix all
three test methods pass, including positive tool/JSON with and without reasoning,
wrong JSON value/key and unknown tool negatives.

Source: patches/qwen3_engine_tool_parser.py. Canary uses an isolated package
view under a task-owned /tmp directory: normal vLLM modules link to the installed
package, while the candidate parser and its bytecode directory are private. The
current backend's argv/checkpoint/auth are reused in memory, with GPU0/private
port9020 and PYTHONPATH selecting the candidate. Do not persist credentials,
change manager routing, or install the candidate into the shared environment
before valid same-case Consumer cold + same-session resume acceptance.

After acceptance, verify and preserve one backup of the original shared parser,
install only this parser, and let existing backends follow their normal idle
unload lifecycle. Do not use admin_switch: its stop operation does not atomically
drain new admissions. After natural unload, a read-only probe wakes the installed
version through the existing manager, and verifies new PID/start time, health,
parser hash and final results. No manager/runtime configuration changes or forced
inference interruption. Stop and clean only the private canary process/tree after
acceptance. API429 and G1 remain out of scope.
