"""Legacy PaperMonitor AUTO-BREAKEVEN must not run — trade_mgmt is sole engine."""
import inspect
import os
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services.trade_management import (
    classify_and_close_paper,
    effective_trade_mgmt_cfg,
    reclassify_legacy_autobe_paper_closes,
)


class TestLegacyAutobeRemoval(unittest.TestCase):
    def test_evaluate_has_no_legacy_autobe_block(self):
        from app.services.strategy_executor import _evaluate_paper_position_against_candles

        src = inspect.getsource(_evaluate_paper_position_against_candles)
        self.assertNotIn("AUTO-BREAKEVEN", src)
        self.assertNotIn("breakeven_at_pct", src)
        self.assertNotIn("breakeven_pct", src)
        self.assertNotIn("trail_enabled", src)
        self.assertNotIn("partial_close_pct", src)

    def test_effective_cfg_ignores_legacy_pct_keys(self):
        cfg = {
            "exit": {
                "breakeven_at_pct": 0,
                "breakeven_pct": 50,
            },
        }
        tm = effective_trade_mgmt_cfg(cfg)
        self.assertFalse(tm["breakeven_enabled"])
        self.assertEqual(tm["breakeven_trigger_pips"], 20.0)

    def test_effective_cfg_enables_on_pips_only(self):
        cfg = {"exit": {"breakeven_at_pips": 15}}
        tm = effective_trade_mgmt_cfg(cfg)
        self.assertTrue(tm["breakeven_enabled"])
        self.assertEqual(tm["breakeven_trigger_pips"], 15.0)

    def test_classify_and_close_paper_uses_shared_classifier(self):
        ex = MagicMock()
        ex.id = 1
        ex.symbol = "XAUUSD"
        ex.direction = "LONG"
        ex.entry_price = 4200.0
        ex.tp_price = 4220.0
        ex.sl_price = 4190.0
        ex.current_sl = 4190.0
        ex.breakeven_applied = False
        ex.notes = "orig_sl=4190.0"
        db = MagicMock()
        with patch(
            "app.services.strategy_executor._close_paper_execution",
        ) as mock_close:
            classify_and_close_paper(ex, "WIN", 4192.0, db, hit_kind="tp")
            mock_close.assert_called_once()
            _args, kwargs = mock_close.call_args
            self.assertNotEqual(kwargs.get("close_label"), "TP hit")

    def test_reclassify_legacy_autobe_flat_win(self):
        class _Ex:
            id = 6299
            symbol = "XAUUSD"
            direction = "LONG"
            entry_price = 4203.275
            exit_price = 4203.275
            pnl_pct = 0.0
            outcome = "WIN"
            notes = "be_moved | TP hit · +0.0% · exit 4203.275"
            strategy_id = 7
            breakeven_applied = True

        ex = _Ex()
        session = MagicMock()
        q = MagicMock()
        q.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [ex]
        session.query.return_value = q
        with patch("app.services.strategy_executor._update_performance"):
            n = reclassify_legacy_autobe_paper_closes(session, hours=168, limit=10)
        self.assertEqual(n, 1)
        self.assertEqual(ex.outcome, "BREAKEVEN")


if __name__ == "__main__":
    unittest.main()
