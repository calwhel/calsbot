"""MFE/MAE excursion tracking and profit telemetry messages."""
import unittest
from unittest.mock import MagicMock

from app.services.strategy_executor import _fmt_close_card, _fmt_excursion_telemetry_line
from app.services.trade_management import update_excursion_pips


class TestExcursionTracking(unittest.TestCase):
    def test_update_excursion_keeps_max_mfe_and_mae(self):
        ex = MagicMock()
        ex.mfe_pips = 10.0
        ex.mae_pips = 2.0
        session = MagicMock()

        changed = update_excursion_pips(ex, 15.5, session)
        self.assertTrue(changed)
        self.assertEqual(ex.mfe_pips, 15.5)
        self.assertEqual(ex.mae_pips, 2.0)
        session.commit.assert_called()

        session.reset_mock()
        changed = update_excursion_pips(ex, -4.2, session)
        self.assertTrue(changed)
        self.assertEqual(ex.mfe_pips, 15.5)
        self.assertEqual(ex.mae_pips, 4.2)

    def test_close_card_includes_excursion_line(self):
        line = _fmt_excursion_telemetry_line(42.0, 3.0)
        self.assertIn("Peak +42", line)
        self.assertIn("Worst −3", line)

        text = _fmt_close_card(
            "Gold Scalp", "XAUUSD", "SHORT",
            entry=4100.0, exit_price=4097.0, outcome="BREAKEVEN",
            pnl_pct=0.0, leverage=1,
            mfe_pips=42.0, mae_pips=3.0,
        )
        self.assertIn("Peak +42", text)
        self.assertIn("Worst −3", text)


class TestTpTuningStat(unittest.TestCase):
    def test_tp_tuning_hint_when_mfe_below_tp(self):
        from app.services.execution_analytics import compute_tp_tuning_stat

        ex1 = MagicMock(outcome="LOSS", mfe_pips=30.0)
        ex2 = MagicMock(outcome="BREAKEVEN", mfe_pips=36.0)
        ex3 = MagicMock(outcome="LOSS", mfe_pips=33.0)
        cfg = {"exit": {"take_profit_pips": 60}}
        stat = compute_tp_tuning_stat([ex1, ex2, ex3], cfg)
        self.assertIsNotNone(stat)
        self.assertEqual(stat["tp_distance_pips"], 60.0)
        self.assertIn("consider a closer target", stat["hint"] or "")


if __name__ == "__main__":
    unittest.main()
