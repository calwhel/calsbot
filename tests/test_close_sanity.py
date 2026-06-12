"""Phantom TP/SL close sanity guards."""
import unittest

from app.services.trade_management import (
    check_directional_exit_hit,
    close_note_label,
    validate_close_sanity,
)


class _Ex:
    id = 1
    symbol = "XAUUSD"
    direction = "SHORT"
    entry_price = 4190.0
    tp_price = 4172.84
    sl_price = 4195.0
    current_sl = 4195.0
    breakeven_applied = False
    notes = "orig_sl=4195.0"


class TestCloseSanity(unittest.TestCase):
    def test_short_tp_requires_price_below_tp(self):
        ex = _Ex()
        self.assertIsNone(check_directional_exit_hit(ex, 4192.73, 4192.73))

    def test_short_tp_hit_when_below(self):
        ex = _Ex()
        hit = check_directional_exit_hit(ex, 4170.0, 4170.0)
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "WIN")

    def test_phantom_tp_relabeled(self):
        ex = _Ex()
        outcome, label = validate_close_sanity(ex, "WIN", 4192.73, "tp")
        self.assertNotEqual(label, "TP hit")

    def test_breakeven_label_not_sl_hit(self):
        ex = _Ex()
        ex.breakeven_applied = True
        ex.current_sl = 4190.0
        ex.sl_price = 4190.0
        label = close_note_label(ex, "BREAKEVEN", 4190.0, "sl")
        self.assertEqual(label, "breakeven stop")

    def test_win_at_entry_downgraded_when_not_profitable(self):
        ex = _Ex()
        ex.breakeven_applied = True
        ex.current_sl = 4190.1
        ex.sl_price = 4190.1
        outcome, label = validate_close_sanity(ex, "WIN", 4190.1, "sl")
        self.assertEqual(outcome, "BREAKEVEN")
        self.assertEqual(label, "breakeven stop")

    def test_phantom_win_at_entry_without_be_is_loss(self):
        ex = _Ex()
        outcome, label = validate_close_sanity(ex, "WIN", 4190.0, "tp")
        self.assertNotEqual(outcome, "WIN")


if __name__ == "__main__":
    unittest.main()
