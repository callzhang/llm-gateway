from __future__ import annotations

import gzip
import importlib.util
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "context_fit_stats.py"
SPEC = importlib.util.spec_from_file_location("context_fit_stats", SCRIPT)
assert SPEC and SPEC.loader
stats = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stats)

START = "10:00:00 INFO     [mgr.context_fit] context_fit: limits for {'qwen3.8-27b': 131072}"
REQ = "10:00:01 INFO     [mgr.router] Sticky chat/completions: task_id=h:a fresh → slot 0 (msgs=2, chars≈10)"
RETRY = "10:00:02 INFO     [mgr.router] Sticky chat/completions: task_id=h:a hit → slot 0 (msgs=2, chars≈10) (retry 1)"
THINK = "10:01:00 INFO     [mgr.context_fit] context_fit: shrank thinking for qwen3.8-27b (input tokens 150000 → 120000, ctx=131072, requested output=32768)"
DEDUP = "10:01:00 INFO     [mgr.context_fit] context_fit: shrank near-duplicates for qwen3.8-27b (input tokens 120000 → 110000, ctx=131072, requested output=32768)"
DROP = "10:01:01 WARNING  [mgr.context_fit] context_fit: dropped 3 oldest turn(s) for qwen3.8-27b (input tokens 110000 → 95000, messages 40 → 30, ctx=131072, requested output=32768)"
OTHER_THINK = "10:05:00 INFO     [mgr.context_fit] context_fit: shrank thinking for qwen3.8-27b (input tokens 101000 → 90000, ctx=131072, requested output=32768)"
CAP = "10:06:00 INFO     [mgr.context_fit] context_fit: capped output 32768 → 30000 for qwen3.8-27b (input tokens=98000, ctx=131072)"


class SummarizeTests(unittest.TestCase):
    def test_chained_steps_are_one_request_classified_by_the_furthest_stage(self):
        s = stats.summarize([START, REQ, THINK, DEDUP, DROP, REQ, OTHER_THINK, CAP])
        self.assertEqual({"drop-turns": 1, "thinking": 1}, dict(s["stages"]))
        self.assertEqual(2, s["requests"])
        self.assertEqual(1, s["capped"])

    def test_retries_are_not_counted_as_new_requests(self):
        self.assertEqual(1, stats.summarize([START, REQ, RETRY])["requests"])

    def test_lines_before_the_first_start_marker_are_ignored(self):
        s = stats.summarize([REQ, THINK, START, REQ])
        self.assertEqual(1, s["requests"])
        self.assertEqual({}, dict(s["stages"]))

    def test_no_marker_means_context_fit_never_ran(self):
        s = stats.summarize([REQ, THINK])
        self.assertFalse(s["started"])
        self.assertIn("has not run", stats.render(s))

    def test_render_reports_shares_of_all_routed_requests(self):
        s = stats.summarize([START] + [REQ] * 3 + [THINK, REQ])
        text = stats.render(s)
        self.assertIn("chat requests routed: 4", text)
        self.assertIn("had to shrink: 1 (25.0000%)", text)


class LogFilesTests(unittest.TestCase):
    def test_rotated_files_are_read_oldest_first_including_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "model_manager.log").write_text("today\n")
            (d / "model_manager.log.1").write_text("yesterday\n")
            with gzip.open(d / "model_manager.log.2.gz", "wt") as f:
                f.write("two days ago\n")
            (d / "model_manager.log.10.gz").write_bytes(gzip.compress(b"ten days ago\n"))
            (d / "model_manager.log.bak").write_text("not a rotation\n")
            lines = [l.strip() for l in stats.read_lines(stats.log_files(d))]
        self.assertEqual(["ten days ago", "two days ago", "yesterday", "today"], lines)


if __name__ == "__main__":
    unittest.main()
