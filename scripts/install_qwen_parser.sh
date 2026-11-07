#!/usr/bin/env bash
# Install patches/qwen3_engine_tool_parser.py into the vLLM that serves the Qwen
# models (--tool-call-parser qwen3_coder resolves to this file).  The patch is
# written against vLLM 0.29, so it is first checked by tests_vllm/ under that
# vLLM's own interpreter; the stock file is kept once as .orig.  Running
# backends keep the parser they loaded: the new one takes effect on the next
# spawn.  See docs/qwen-native-output-grammar-fix.md.
#
#   VLLM_BIN=/path/to/env/bin/vllm scripts/install_qwen_parser.sh
set -euo pipefail

GATEWAY_DIR=$(cd "$(dirname "$0")/.." && pwd)
VLLM_BIN=${VLLM_BIN:?set VLLM_BIN to the vllm executable the backends run}
PY=$(dirname "$VLLM_BIN")/python
SRC=$GATEWAY_DIR/patches/qwen3_engine_tool_parser.py
# vLLM logs to stdout on import, so the path is the last line.
DEST=$("$PY" -c 'import os, vllm.tool_parsers as t; print(os.path.join(os.path.dirname(t.__file__), "qwen3_engine_tool_parser.py"))' | tail -n 1)

(cd "$GATEWAY_DIR" && "$PY" -m unittest discover -s tests_vllm -q)

if ! cmp -s "$SRC" "$DEST"; then
    [ -e "$DEST.orig" ] || cp -p "$DEST" "$DEST.orig"
    install -m 0664 "$SRC" "$DEST"
fi
cmp "$SRC" "$DEST"
echo "installed $(sha256sum "$DEST" | cut -c1-12) -> $DEST"
