"""Source tests for standalone gemini-gold runner spawn + diagnostics."""
from __future__ import annotations

import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "app" / "gemini_gold_runner.py").read_text()
START_SH = (ROOT / "start.sh").read_text()


class GeminiGoldRunnerSourceTests(unittest.TestCase):
    def test_runner_has_early_stdout_diagnostics(self):
        self.assertIn("__main__ entry", RUNNER)
        self.assertIn("runner main() pid=", RUNNER)
        self.assertIn("flush=True", RUNNER)

    def test_start_sh_spawn_has_post_sleep_and_preflight(self):
        self.assertIn("app.gemini_gold_runner", START_SH)
        self.assertIn("GEMINI_GOLD_STANDALONE", START_SH)
        self.assertIn("gemini-gold post-sleep", START_SH)
        self.assertIn("gemini-gold FATAL", START_SH)
        self.assertIn("python3 -u -m app.gemini_gold_runner", START_SH)
        self.assertNotIn("exec python3 -m app.gemini_gold_runner", START_SH)


if __name__ == "__main__":
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    unittest.main()
