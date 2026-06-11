"""Paper breakeven must persist current_sl before notify; classify SL hits correctly."""
import unittest
from unittest.mock import MagicMock

from app.services.trade_management import (
    active_sl,
    breakeven_sl_price,
    classify_sl_close_outcome,
    persist_paper_stop_level,
)


class _Ex:
    id = 99
    symbol = "XAUUSD"
    direction = "SHORT"
    entry_price = 4077.57
    sl_price = 4077.57
    current_sl = 4087.57
    breakeven_applied = True
    notes = "orig_sl=4087.57 | be_moved"


class TestBreakevenSlPaper(unittest.TestCase):
    def test_active_sl_prefers_current_sl(self):
        ex = _Ex()
        self.assertEqual(active_sl(ex), 4087.57)

    def test_short_breakeven_below_entry(self):
        sl = breakeven_sl_price("XAUUSD", 4077.57, "SHORT", offset_pips=1.0)
        self.assertLess(sl, 4077.57)

    def test_mislabeled_breakeven_at_original_sl_is_loss(self):
        ex = _Ex()
        outcome = classify_sl_close_outcome(ex, 4087.57)
        self.assertEqual(outcome, "LOSS")

    def test_true_breakeven_at_entry(self):
        ex = _Ex()
        ex.current_sl = 4077.57
        outcome = classify_sl_close_outcome(ex, 4077.57)
        self.assertEqual(outcome, "BREAKEVEN")

    def test_persist_paper_stop_updates_both_columns(self):
        ex = _Ex()
        ex.current_sl = 4087.57
        ex.sl_price = 4087.57
        ex.breakeven_applied = False
        session = MagicMock()
        session.refresh = MagicMock()
        ok = persist_paper_stop_level(ex, 4077.57, session, mark_breakeven=True)
        self.assertTrue(ok)
        self.assertEqual(ex.current_sl, 4077.57)
        self.assertEqual(ex.sl_price, 4077.57)
        self.assertTrue(ex.breakeven_applied)
        session.commit.assert_called_once()


if __name__ == "__main__":
    unittest.main()
