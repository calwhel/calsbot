"""Tests for cTrader kline staleness detection and per-cycle sweep."""
import time
import unittest
from unittest import mock

from app.services import ctrader_price_feed as feed


class TestKlineStaleness(unittest.TestCase):
    def setUp(self):
        feed._kline_cache.clear()
        feed._last_kline_update.clear()
        feed._last_spot_tick_mono = time.monotonic()

    def test_bar_timestamp_stale_while_ticks_flow(self):
        old_ts_ms = int((time.time() - 3600) * 1000)
        rows = [[old_ts_ms, 1.0, 1.0, 1.0, 1.0, 0.0]]
        stale, detail = feed._kline_cache_is_stale("XAUUSD", rows, "15m")
        self.assertTrue(stale)
        self.assertIn("bar_ts=", detail)

    def test_drift_stale_while_ticks_flow(self):
        now_ms = int(time.time() * 1000)
        rows = [[now_ms, 4088.0, 4088.0, 4088.0, 4088.82, 0.0]]
        with mock.patch.object(feed, "get_price", return_value=4147.10):
            stale, detail = feed._kline_cache_is_stale("XAUUSD", rows, "15m")
        self.assertTrue(stale)
        self.assertIn("drift=", detail)

    def test_fresh_klines_not_stale(self):
        now_ms = int(time.time() * 1000)
        rows = [[now_ms, 4147.0, 4147.0, 4147.0, 4147.0, 0.0]]
        feed._last_kline_update["XAUUSD"] = time.monotonic()
        with mock.patch.object(feed, "get_price", return_value=4147.0):
            stale, _ = feed._kline_cache_is_stale("XAUUSD", rows, "15m")
        self.assertFalse(stale)


class TestKlineStalenessSweep(unittest.IsolatedAsyncioTestCase):
    async def test_sweep_rebuilds_on_drift(self):
        feed._kline_cache.clear()
        feed._last_spot_tick_mono = time.monotonic()
        now_ms = int(time.time() * 1000)
        feed._kline_cache[("XAUUSD", "15m", 80)] = (
            [[now_ms, 4088.0, 4088.0, 4088.0, 4088.82, 0.0]],
            time.monotonic(),
        )
        with mock.patch.object(feed, "_PROTO_OK", True), \
             mock.patch.object(feed, "get_price", return_value=4147.10), \
             mock.patch.object(feed, "get_klines", new_callable=mock.AsyncMock) as m_get, \
             mock.patch.object(feed, "_invalidate_tb_conn", new_callable=mock.AsyncMock):
            n = await feed.sweep_stale_klines(symbols=["XAUUSD"], timeframes=["15m"])
        self.assertEqual(n, 1)
        m_get.assert_awaited()


if __name__ == "__main__":
    unittest.main()
