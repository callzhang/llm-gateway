#!/usr/bin/env bash
for name in qwen36_27b qwen36_35b; do
  pid_file="/home/derek/Projects/llm-gateway/${name}.pid"
  if [[ -f "$pid_file" ]]; then
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" && echo "$name stopped (PID $pid)" || echo "$name kill failed"
    else
      echo "$name not running"
    fi
    rm -f "$pid_file"
  else
    echo "$name pid file not found"
  fi
done
