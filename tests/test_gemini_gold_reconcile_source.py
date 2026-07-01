"""Gemini Gold broker reconcile wiring — source regression tests."""
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOOP = (ROOT / "app" / "gemini_gold_trader" / "loop.py").read_text()
ROUTES = (ROOT / "app" / "gemini_gold_trader" / "routes.py").read_text()
TELEGRAM = (ROOT / "app" / "gemini_gold_trader" / "telegram_notify.py").read_text()


class GeminiGoldReconcileSourceTests(unittest.TestCase):
    def test_background_loop_runs_broker_reconcile_every_cycle(self):
        self.assertIn("async def _sync_closed_outcomes_pass() -> None:", LOOP)
        self.assertIn("from app.services.strategy_executor import _reconcile_forex_closes", LOOP)
        self.assertIn("_reconcile_forex_closes(user_id=demo_uid)", LOOP)
        self.assertIn("_call_with_db_session(sync_closed_trade_notifications", LOOP)
        self.assertRegex(
            LOOP,
            re.compile(
                r"await asyncio\.wait_for\(run_gemini_gold_trader_loop\(\).*?"
                r"await asyncio\.wait_for\(_sync_closed_outcomes_pass\(\)",
                re.S,
            ),
        )

    def test_startup_reconcile_before_scan_loop(self):
        self.assertIn("[gemini-gold] startup broker reconcile pass complete", LOOP)
        self.assertRegex(
            LOOP,
            re.compile(
                r"startup broker reconcile pass complete.*?while True:",
                re.S,
            ),
        )

    def test_status_routes_schedule_background_reconcile(self):
        self.assertIn("_schedule_status_background_reconcile", ROUTES)
        self.assertIn("_reconcile_forex_closes(user_id=uid)", ROUTES)
        self.assertIn("open_executions", ROUTES)
        self.assertIn("open_slots_used", ROUTES)

    def test_manual_reconcile_endpoint(self):
        self.assertIn('@router.post("/api/gemini-gold-trader/reconcile")', ROUTES)
        self.assertIn("open_before", ROUTES)
        self.assertIn("open_after", ROUTES)
        self.assertIn("reconcile_orphan_open_executions", ROUTES)
        self.assertIn("orphan_reconcile", ROUTES)

    def test_loop_runs_orphan_reconcile(self):
        self.assertIn("reconcile_orphan_open_executions", LOOP)

    def test_close_notifications_not_gated_by_telegram_for_outcome_sync(self):
        self.assertIn("async def sync_closed_trade_notifications", TELEGRAM)
        self.assertIn("was_new = record_outcome_from_execution(db, did, ex)", TELEGRAM)
        self.assertIn("if not notifications_enabled:", TELEGRAM)


if __name__ == "__main__":
    unittest.main()
