#!/usr/bin/env python3
"""Report how often context_fit had to touch a request, and how far it went.

Reads model_manager's log (rotated files oldest first) from the first
"context_fit: limits for" line on, counts the chat requests it routed, and groups
the context_fit lines into one chain per fitted request: consecutive shrink
steps belong to the same request when each step starts at the token count the
previous one ended at.  A chain is classified by the furthest stage it reached.

    context_fit_stats.py [--log-dir DIR]
"""
from __future__ import annotations

import argparse
import gzip
import re
from collections import Counter
from pathlib import Path

START = "context_fit: limits for"
REQUEST_LINES = ("[mgr.router] Sticky chat/completions", "[mgr.router] chat/completions no task_id")
STEP = re.compile(
    r"context_fit: (?:shrank (?P<what>thinking|near-duplicates|tool arguments/results)"
    r"|dropped \d+ oldest turn\(s\)) for (?P<model>\S+) \(input tokens (?P<before>\d+) → (?P<after>\d+)"
)
STAGE_OF = {"thinking": "thinking", "near-duplicates": "dedup", "tool arguments/results": "tool-cap", None: "drop-turns"}
RANK = ["cap-output", "thinking", "dedup", "tool-cap", "drop-turns"]


def log_files(log_dir: Path) -> list[Path]:
    """model_manager.log.N[.gz] ... model_manager.log, oldest first."""
    rotated = [p for p in log_dir.glob("model_manager.log.*") if p.name.split(".log.")[1].split(".")[0].isdigit()]
    rotated.sort(key=lambda p: -int(p.name.split(".log.")[1].split(".")[0]))
    return rotated + [log_dir / "model_manager.log"]


def read_lines(paths: list[Path]):
    for path in paths:
        if not path.exists():
            continue
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", errors="replace") as f:
            yield from f


def summarize(lines) -> dict:
    """Counts from the first START line on; None-valued 'since' when never seen."""
    started = False
    requests = 0
    chains: list[dict] = []          # {"model", "after", "stage"}
    capped = no_room = filler = 0
    for line in lines:
        if not started:
            started = START in line
            if not started:
                continue
        if any(marker in line for marker in REQUEST_LINES) and "(retry" not in line:
            requests += 1
        elif "context_fit: capped output" in line:
            capped += 1
        elif "no room left" in line and "context_fit" in line:
            no_room += 1
        elif "had no user turn" in line and "context_fit" in line:
            filler += 1
        else:
            step = STEP.search(line)
            if not step:
                continue
            stage = STAGE_OF[step["what"]]
            before, after = int(step["before"]), int(step["after"])
            last = chains[-1] if chains else None
            if last and last["model"] == step["model"] and last["after"] == before:
                last["after"] = after
                if RANK.index(stage) > RANK.index(last["stage"]):
                    last["stage"] = stage
            else:
                chains.append({"model": step["model"], "after": after, "stage": stage})
    return {"started": started, "requests": requests, "capped": capped, "no_room": no_room,
            "filler": filler, "stages": Counter(c["stage"] for c in chains)}


def render(s: dict) -> str:
    if not s["started"]:
        return "no 'context_fit: limits for' line in the logs: model_manager has not run context_fit yet"
    n = s["requests"]
    pct = lambda x: f"{x / n:.4%}" if n else "n/a"
    rows = [f"chat requests routed: {n}"]
    shrunk = sum(s["stages"].values())
    rows.append(f"requests whose input had to shrink: {shrunk} ({pct(shrunk)})")
    for stage in RANK[1:]:
        rows.append(f"  furthest stage {stage:<10} {s['stages'][stage]:>5}  ({pct(s['stages'][stage])})")
    rows.append(f"output cap lowered (any request): {s['capped']} ({pct(s['capped'])})")
    rows.append(f"left to vLLM (no room even for the latest turn): {s['no_room']}")
    rows.append(f"filler user turn added: {s['filler']}")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log-dir", type=Path, default=Path(__file__).resolve().parents[1] / "logs")
    args = parser.parse_args()
    print(render(summarize(read_lines(log_files(args.log_dir)))))


if __name__ == "__main__":
    main()
