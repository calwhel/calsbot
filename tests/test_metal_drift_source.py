"""Source-aware metal kline/live drift limits."""
import unittest
from unittest.mock import AsyncMock, patch

from app.services import tradfi_prices as tp
from app.services.strategy_executor import _check_time_filter


class TestMetalKlineDrift(unittest.TestCase):
    def test_spot_source_uses_looser_cap(self):
        self.assertEqual(tp.metal_kline_drift_limit("kraken"), tp.METAL_SPOT_KLINE_MAX_DRIFT_PCT)
        self.assertEqual(tp.metal_kline_drift_limit("coinbase"), tp.METAL_SPOT_KLINE_MAX_DRIFT_PCT)

    def test_unknown_source_uses_strict_cap(self):
        self.assertEqual(tp.metal_kline_drift_limit(None), tp.METAL_KLINE_LIVE_MAX_DRIFT_PCT)
        self.assertEqual(tp.metal_kline_drift_limit("yahoo"), tp.METAL_KLINE_LIVE_MAX_DRIFT_PCT)

    def test_kraken_paxg_drift_within_spot_cap(self):
        """PAXG vs XAUUSD ~0.29% must pass the spot-aligned threshold."""
        self.assertGreaterEqual(tp.METAL_SPOT_KLINE_MAX_DRIFT_PCT, 0.29)

    def test_source_cached_after_fetch(self):
        tp._METAL_KLINE_SOURCE_CACHE.clear()
        tp._METAL_KLINE_SOURCE_CACHE[("XAUUSD", "15m", 80)] = ("kraken", __import__("datetime").datetime.utcnow())
        self.assertEqual(tp.get_metal_kline_source("XAUUSD", "15m", 80), "kraken")


class TestKillzoneSessionFilter(unittest.TestCase):
    def test_london_kz_inside_window(self):
        with patch("app.services.strategy_executor.datetime") as mock_dt:
            mock_dt.utcnow.return_value = __import__("datetime").datetime(2026, 6, 8, 8, 0)
            ok = _check_time_filter({"session": {"sessions": ["london_kz"]}})
        self.assertTrue(ok)

    def test_london_kz_outside_window(self):
        with patch("app.services.strategy_executor.datetime") as mock_dt:
            mock_dt.utcnow.return_value = __import__("datetime").datetime(2026, 6, 8, 11, 0)
            ok = _check_time_filter({"session": {"sessions": ["london_kz"]}})
        self.assertFalse(ok)

    def test_any_kz_matches_ny_window(self):
        with patch("app.services.strategy_executor.datetime") as mock_dt:
            mock_dt.utcnow.return_value = __import__("datetime").datetime(2026, 6, 8, 13, 0)
            ok = _check_time_filter({"session": {"sessions": ["any_kz"]}})
        self.assertTrue(ok)


class TestMetalLiveFetchSource(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_stores_kline_source(self):
        tp._METAL_KLINE_SOURCE_CACHE.clear()
        with patch.object(
            tp, "_fetch_coinbase_metals_klines",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_kraken_metals_klines",
            new_callable=AsyncMock,
            return_value=[
                [1_700_000_000_000, 4320, 4330, 4310, 4328.0, 1.0]
            ] * 80,
        ), patch.object(
            tp, "_fetch_fmp_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch(
            "app.services.ctrader_price_feed.is_live", return_value=False,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "15m", 80)
        self.assertEqual(len(rows), 80)
        self.assertEqual(tp.get_metal_kline_source("XAUUSD", "15m", 80), "kraken")


if __name__ == "__main__":
    unittest.main()
