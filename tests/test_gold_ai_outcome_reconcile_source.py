from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
TELEGRAM_NOTIFY = (ROOT / "app" / "gold_ai_trader" / "telegram_notify.py").read_text()


class GoldAiOutcomeReconcileSourceTests(unittest.TestCase):
    def test_reconcile_not_gated_by_telegram_toggle(self):
        self.assertNotIn(
            "if not telegram_notifications_enabled():\n        return 0",
            TELEGRAM_NOTIFY,
        )
        self.assertIn(
            "notifications_enabled = telegram_notifications_enabled()",
            TELEGRAM_NOTIFY,
        )
        self.assertRegex(
            TELEGRAM_NOTIFY,
            re.compile(
                r"was_new = record_outcome_from_execution\(db, did, ex\).*?"
                r"if not was_new:\s+continue.*?"
                r"if not notifications_enabled:\s+continue.*?"
                r"ok = await notify_trade_close\(",
                re.S,
            ),
        )

    def test_decision_id_fallback_uses_conditions_met(self):
        self.assertIn("def _decision_id_from_execution(execution) -> Optional[int]:", TELEGRAM_NOTIFY)
        self.assertIn("meta = getattr(execution, \"conditions_met\", None)", TELEGRAM_NOTIFY)
        self.assertIn("raw = meta.get(\"gold_ai_decision_id\")", TELEGRAM_NOTIFY)


if __name__ == "__main__":
    unittest.main()
