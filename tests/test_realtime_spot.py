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

    async def test_fetch_parallel_ctrader_on_demand_beats_fmp(self):
        with patch.object(rs, "_ctrader_fresh", return_value=None), patch.object(
            rs, "read_fresh_cached", return_value=None
        ), patch.object(
            rs, "_fetch_ctrader_on_demand",
            new_callable=AsyncMock,
            return_value=(28786.0, "ctrader"),
        ), patch.object(
            rs, "_fetch_fmp_on_demand",
            new_callable=AsyncMock,
            return_value=(28780.0, "fmp"),
        ), patch.object(
            rs, "_fetch_yfinance_spot", new_callable=AsyncMock, return_value=None
        ):
            hit = await rs.fetch_parallel("NAS100", "index")
        self.assertEqual(hit, (28786.0, "ctrader"))

    async def test_fetch_ctrader_on_demand_waits_for_stream(self):
        with patch("app.services.ctrader_price_feed.broker_session_ready", return_value=True), patch(
            "app.services.ctrader_price_feed.is_live", return_value=True
        ), patch.object(
            rs, "_read_ctrader", side_effect=[None, None, (1.0851, "ctrader")]
        ), patch.object(rs, "_CTRADER_ON_DEMAND_WAIT_S", 1.0):
            hit = await rs._fetch_ctrader_on_demand(
                "EURUSD", "forex", max_age=5.0,
            )
        self.assertEqual(hit, (1.0851, "ctrader"))

    async def test_fetch_ctrader_on_demand_trendbar_close(self):
        import time as _time
        bar_ts = int((_time.time() - 65) * 1000)
        with patch("app.services.ctrader_price_feed.broker_session_ready", return_value=True), patch(
            "app.services.ctrader_price_feed.is_live", return_value=False
        ), patch.object(rs, "_CTRADER_ON_DEMAND_WAIT_S", 0.0), patch(
            "app.services.ctrader_price_feed.get_klines",
            new_callable=AsyncMock,
            return_value=[[bar_ts, 1.0, 1.1, 0.9, 1.0850, 0.0]],
        ), patch.object(rs, "_persist_tick") as mock_store:
            hit = await rs._fetch_ctrader_on_demand(
                "EURUSD", "forex", max_age=15.0, paper_ok=True,
            )
        self.assertEqual(hit, (1.0850, "ctrader"))
        mock_store.assert_called_once_with("EURUSD", 1.0850, "ctrader")

    async def test_fetch_parallel_twelve_data_fire_time_only(self):
        with patch.object(rs, "_ctrader_fresh", return_value=None), patch.object(
            rs, "read_fresh_cached", return_value=None
        ), patch.object(
            rs, "_fetch_fmp_on_demand", new_callable=AsyncMock, return_value=None
        ), patch.object(
            rs, "_fetch_twelve_data_on_demand",
            new_callable=AsyncMock,
            return_value=(28786.0, "twelvedata"),
        ) as mock_td, patch.object(
            rs, "_fetch_yfinance_spot", new_callable=AsyncMock, return_value=None
        ):
            hit = await rs.fetch_parallel(
                "NAS100", "index", twelve_data_ok=True,
            )
        self.assertEqual(hit, (28786.0, "twelvedata"))
        mock_td.assert_awaited_once()

    async def test_fetch_parallel_skips_twelve_data_without_flag(self):
        with patch.object(rs, "_ctrader_fresh", return_value=None), patch.object(
            rs, "read_fresh_cached", return_value=None
        ), patch.object(
            rs, "_fetch_fmp_on_demand", new_callable=AsyncMock, return_value=None
        ), patch.object(
            rs, "_fetch_twelve_data_on_demand", new_callable=AsyncMock,
        ) as mock_td, patch.object(
            rs, "_fetch_yfinance_spot",
            new_callable=AsyncMock,
            return_value=(28780.0, "yfinance"),
        ):
            hit = await rs.fetch_parallel("NAS100", "index", paper_ok=True)
        mock_td.assert_not_awaited()
        self.assertEqual(hit, (28780.0, "yfinance"))

    async def test_fetch_metals_parallel_stores_best(self):
        with patch.object(
            rs, "_fetch_coinbase", new_callable=AsyncMock, return_value=2649.5
        ), patch.object(rs, "_fetch_kraken", new_callable=AsyncMock, return_value=2650.0), patch.object(
            rs, "_persist_tick"
        ) as mock_store:
            hit = await rs._fetch_metals_parallel("XAUUSD")
        self.assertEqual(hit, (2649.5, "coinbase"))
        mock_store.assert_called_once_with("XAUUSD", 2649.5, "coinbase")


if __name__ == "__main__":
    unittest.main()
