"""Unified real-time spot resolver."""
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import realtime_spot as rs


class TestRealtimeSpot(unittest.TestCase):
    def test_max_age_metals_stricter_than_forex(self):
        self.assertLess(rs.max_age_s("XAUUSD", "forex"), rs.max_age_s("EURUSD", "forex") + 1)
        self.assertEqual(rs.max_age_s("XAUUSD", "forex"), rs._MAX_AGE_METALS)

    def test_read_fresh_cached_picks_ctrader_over_binance(self):
        with patch.object(rs, "_read_ctrader", return_value=(2650.0, "ctrader")), patch.object(
            rs, "_read_store", return_value=(2649.0, "binance")
        ), patch.object(rs, "_read_metals_cache", return_value=None), patch.object(
            rs, "_read_fmp_fresh", return_value=None
        ):
            hit = rs.read_fresh_cached("XAUUSD", "forex")
        self.assertEqual(hit, (2650.0, "ctrader"))

    def test_read_fresh_cached_rejects_stale_metals_cache(self):
        with patch("app.services.metals_spot_feed._PRICE_CACHE", {
            "XAUUSD": (2650.0, datetime.utcnow() - timedelta(seconds=30)),
        }):
            hit = rs._read_metals_cache("XAUUSD", max_age=3.0)
        self.assertIsNone(hit)

    def test_get_realtime_spot_returns_cached_without_fetch(self):
        with patch.object(
            rs, "read_fresh_cached", return_value=(1.0850, "ctrader")
        ), patch.object(rs, "fetch_parallel", new_callable=AsyncMock) as mock_fetch:
            import asyncio
            px = asyncio.run(rs.get_realtime_spot("EURUSD", "forex"))
        self.assertEqual(px, 1.0850)
        mock_fetch.assert_not_called()

    def test_get_realtime_spot_force_fetch(self):
        with patch.object(
            rs, "read_fresh_cached", return_value=(1.0850, "ctrader")
        ), patch.object(
            rs, "fetch_parallel", new_callable=AsyncMock, return_value=(1.0851, "ctrader")
        ):
            import asyncio
            px = asyncio.run(rs.get_realtime_spot("EURUSD", "forex", force_fetch=True))
        self.assertEqual(px, 1.0851)


class TestRealtimeSpotAsync(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_parallel_ctrader_beats_yfinance(self):
        with patch.object(rs, "_ctrader_fresh", return_value=(28786.0, "ctrader")), patch.object(
            rs, "read_fresh_cached", return_value=None
        ), patch.object(
            rs, "_fetch_fmp_on_demand", new_callable=AsyncMock, return_value=None
        ), patch.object(
            rs, "_fetch_yfinance_spot",
            new_callable=AsyncMock,
            return_value=(28780.0, "yfinance"),
        ):
            hit = await rs.fetch_parallel("NAS100", "index")
        self.assertEqual(hit, (28786.0, "ctrader"))

    async def test_fetch_parallel_yfinance_when_ctrader_missing(self):
        with patch.object(rs, "_ctrader_fresh", return_value=None), patch.object(
            rs, "read_fresh_cached", return_value=None
        ), patch.object(
            rs, "_fetch_fmp_on_demand", new_callable=AsyncMock, return_value=None
        ), patch.object(
            rs, "_fetch_yfinance_spot",
            new_callable=AsyncMock,
            return_value=(1.0852, "yfinance"),
        ):
            hit = await rs.fetch_parallel("EURUSD", "forex", paper_ok=True)
        self.assertEqual(hit, (1.0852, "yfinance"))

    async def test_fetch_parallel_fmp_before_yfinance(self):
        with patch.object(rs, "_ctrader_fresh", return_value=None), patch.object(
            rs, "read_fresh_cached", return_value=None
        ), patch.object(
            rs, "_fetch_fmp_on_demand",
            new_callable=AsyncMock,
            return_value=(28790.0, "fmp"),
        ), patch.object(
            rs, "_fetch_yfinance_spot",
            new_callable=AsyncMock,
            return_value=(28780.0, "yfinance"),
        ):
            hit = await rs.fetch_parallel("NAS100", "index")
        self.assertEqual(hit, (28790.0, "fmp"))

    async def test_fetch_metals_parallel_stores_best(self):
        with patch.object(rs, "_fetch_binance", new_callable=AsyncMock, return_value=2650.0), patch.object(
            rs, "_fetch_coinbase", new_callable=AsyncMock, return_value=2649.5
        ), patch.object(rs, "_fetch_kraken", new_callable=AsyncMock, return_value=None), patch.object(
            rs, "_persist_tick"
        ) as mock_store:
            hit = await rs._fetch_metals_parallel("XAUUSD")
        self.assertEqual(hit, (2650.0, "binance"))
        mock_store.assert_called_once_with("XAUUSD", 2650.0, "binance")


if __name__ == "__main__":
    unittest.main()
