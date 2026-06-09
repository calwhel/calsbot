"""Gold/XAUUSD entry price guards — spot vs GC=F futures."""
import unittest
from unittest import mock

from app.services.tradfi_prices import (
    METAL_KLINE_LIVE_MAX_DRIFT_PCT,
    confirm_entry_price,
    is_metal_symbol,
    metal_klines_match_live_spot,
)


class TestMetalEntryPrice(unittest.TestCase):
    def test_is_metal_symbol(self):
        self.assertTrue(is_metal_symbol("XAUUSD"))
        self.assertTrue(is_metal_symbol("xau/usd"))
        self.assertFalse(is_metal_symbol("EURUSD"))

    def test_klines_rejected_when_futures_diverge(self):
        rows = [[0, 2600, 2610, 2590, 2600.0, 0]]
        with mock.patch(
            "app.services.tradfi_prices.get_price_fresh",
            mock.AsyncMock(return_value=2650.0),
        ):
            import asyncio
            ok = asyncio.run(metal_klines_match_live_spot("XAUUSD", rows))
        self.assertFalse(ok)

    def test_klines_accepted_when_aligned(self):
        rows = [[0, 2648, 2652, 2645, 2650.0, 0]]
        with mock.patch(
            "app.services.tradfi_prices.get_price_fresh",
            mock.AsyncMock(return_value=2650.0),
        ):
            import asyncio
            ok = asyncio.run(metal_klines_match_live_spot("XAUUSD", rows))
        self.assertTrue(ok)

    def test_drift_threshold_sane_default(self):
        self.assertLessEqual(METAL_KLINE_LIVE_MAX_DRIFT_PCT, 1.0)

    def test_confirm_entry_rejects_stale_proposed(self):
        with mock.patch(
            "app.services.tradfi_prices.get_price_fresh",
            mock.AsyncMock(return_value=2650.0),
        ):
            import asyncio
            px, reason = asyncio.run(
                confirm_entry_price("XAUUSD", "forex", 2600.0)
            )
        self.assertIsNone(px)
        self.assertIn("drift", reason)

    def test_confirm_entry_accepts_fresh_proposed(self):
        with mock.patch(
            "app.services.tradfi_prices.get_price_fresh",
            mock.AsyncMock(return_value=2650.0),
        ):
            import asyncio
            px, reason = asyncio.run(
                confirm_entry_price("XAUUSD", "forex", 2649.0)
            )
        self.assertEqual(px, 2650.0)
        self.assertIn("confirmed", reason)


if __name__ == "__main__":
    unittest.main()
