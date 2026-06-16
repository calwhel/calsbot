"""Live-fire diagnostic logging helpers."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import inspect
import unittest
from types import SimpleNamespace

from app.services import strategy_executor as se
from app.services.strategy_account_assignments import _format_assignments_log


class TestLiveFireLogging(unittest.TestCase):
    def test_forex_live_blockers_not_approved(self):
        prefs = SimpleNamespace(
            ctrader_access_token="tok",
            ctrader_account_id="12345",
            forex_approved=False,
        )
        blockers = se._forex_live_blockers(prefs, user=None)
        self.assertIn("forex live not approved", blockers)

    def test_forex_live_blockers_admin_bypass(self):
        prefs = SimpleNamespace(
            ctrader_access_token="tok",
            ctrader_account_id="12345",
            forex_approved=False,
        )
        user = SimpleNamespace(is_admin=True)
        self.assertEqual(se._forex_live_blockers(prefs, user=user), [])

    def test_format_fire_targets(self):
        targets = [
            {"ctrader_account_id": "47516246", "lot_size": 0.05},
            {"ctrader_account_id": "47465772", "lot_size": None},
        ]
        self.assertEqual(
            se._format_fire_targets_log(targets),
            "[47516246@0.05, 47465772@default]",
        )
        self.assertEqual(
            _format_assignments_log(targets),
            "[47516246@0.05, 47465772@default]",
        )

    def test_fanout_except_logs_critical(self):
        src = inspect.getsource(se._ctrader_fanout_live_fire)
        self.assertIn("logger.critical", src)
        self.assertIn("[live-fire] fan-out FAILED", src)

    def test_evaluate_and_fire_live_skip_logging(self):
        src = inspect.getsource(se.evaluate_and_fire)
        self.assertIn("[live-fire] SKIPPED strategy=", src)
        self.assertIn("resolved_targets=", src)


if __name__ == "__main__":
    unittest.main()
