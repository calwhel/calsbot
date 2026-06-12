"""Pips P&L computation — paper vs live, sanity guard, display."""
import unittest

from app.services.trade_management import (
    compute_execution_pips_pnl,
    compute_pips_from_prices,
    format_pips_display,
    guard_pips_sanity,
    is_forex_like_asset,
)


class _Ex:
    id = 99
    symbol = "XAUUSD"
    direction = "SHORT"
    entry_price = 4190.0
    exit_price = 4170.0
    asset_class = "index"
    tp1_done = False


class TestPipsPnl(unittest.TestCase):
    def test_index_asset_is_forex_like(self):
        self.assertTrue(is_forex_like_asset("index"))

    def test_short_gold_pips(self):
        pips = compute_pips_from_prices(4190.0, 4170.0, "SHORT", "XAUUSD")
        self.assertEqual(pips, 200.0)

    def test_paper_index_execution_gets_pips(self):
        ex = _Ex()
        pips = compute_execution_pips_pnl(ex, 4170.0)
        self.assertEqual(pips, 200.0)

    def test_null_pips_shows_pending_not_zero(self):
        disp = format_pips_display(
            pips_pnl=None,
            entry=4190.0,
            exit_price=4170.0,
            symbol="XAUUSD",
            direction="SHORT",
            notes="pending_reconcile",
        )
        self.assertEqual(disp, "pending")

    def test_sanity_guard_recomputes_zero(self):
        pips = guard_pips_sanity(1, 4190.0, 4170.0, "XAUUSD", "SHORT", 0.0)
        self.assertEqual(pips, 200.0)


if __name__ == "__main__":
    unittest.main()
