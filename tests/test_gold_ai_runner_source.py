"""Source tests for standalone gold-ai runner architecture."""
from __future__ import annotations

import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LOOP = (ROOT / "app" / "gold_ai_trader" / "loop.py").read_text()
RUNNER = (ROOT / "app" / "gold_ai_runner.py").read_text()
PORTAL_MOUNT = (ROOT / "app" / "gold_ai_trader" / "portal_mount.py").read_text()
ROUTES = (ROOT / "app" / "gold_ai_trader" / "routes.py").read_text()
START_SH = (ROOT / "start.sh").read_text()


class GoldAiRunnerSourceTests(unittest.TestCase):
    def test_lock_machinery_removed_from_loop(self):
        for token in (
            "_LOCK_ID",
            "_lock_conn",
            "_acquire_gold_ai_lock",
            "_reconnect_gold_ai_lock",
            "_release_gold_ai_lock",
            "_reclaim_stale_gold_ai_locks",
            "_ping_lock_connection",
            "_locked_loop_forever",
            "has_local_lock",
            "GOLD_AI_LOCK_RECLAIM_AFTER_MISSES",
            "GOLD_AI_LOCK_RECLAIM_MIN_IDLE_S",
            "GOLD_AI_ON_DEMAND_FORCE_RECLAIM_AFTER_S",
            "forced lock reclaim",
            "advisory lock acquired",
            "advisory lock connection lost",
        ):
            self.assertNotIn(token, LOOP, msg=f"unexpected lock token: {token}")

    def test_standalone_scan_driver_present(self):
        self.assertIn("async def _scan_loop_forever()", LOOP)
        self.assertIn("async def start_gold_ai_trader_loop()", LOOP)
        self.assertIn("def gold_ai_loop_disabled_in_gunicorn()", LOOP)
        self.assertIn("run_with_db", LOOP)
        self.assertIn("async def _call_with_db_session", LOOP)

    def test_gunicorn_and_status_do_not_start_loop(self):
        self.assertIn("DISABLE_GOLD_AI_IN_GUNICORN", PORTAL_MOUNT)
        self.assertIn("loop disabled in gunicorn", PORTAL_MOUNT)
        self.assertNotIn("maybe_start_background_loop()", ROUTES)

    def test_runner_and_start_sh_spawn_subprocess(self):
        self.assertIn("gold_ai_runner", RUNNER)
        self.assertIn("GOLD_AI_STANDALONE", RUNNER)
        self.assertIn("app.gold_ai_runner", START_SH)
        self.assertIn("DISABLE_GOLD_AI_IN_GUNICORN", START_SH)
        self.assertIn("DISABLE_STANDALONE_GOLD_AI", START_SH)

    def test_watchdog_has_no_lock_gate(self):
        self.assertIn("async def _watchdog_loop_forever()", LOOP)
        self.assertIn("await _restart_background_loop(reason)", LOOP)
        self.assertNotIn("force_reclaim", LOOP)


if __name__ == "__main__":
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    unittest.main()
