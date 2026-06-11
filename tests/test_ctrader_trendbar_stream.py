"""Trendbar fetch on live spot stream (no second socket)."""
import asyncio
import time
import unittest
from unittest import mock

from app.services import ctrader_price_feed as feed


class TestTrendbarStreamFetch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        feed._feed_live = False
        feed._stream_writer = None
        feed._stream_reader = None
        feed._stream_ctid = 0
        feed._trendbar_block_until = 0.0
        feed._trendbar_block_reason = None
        feed._symbol_id_map.clear()
        feed._symbol_id_map["XAUUSD"] = 41

    def test_blocked_reason_when_live_without_stream(self):
        feed._feed_live = True
        feed._stream_writer = None
        reason = feed.trendbar_fetch_blocked_reason()
        self.assertIn("stream session", reason or "")

    def test_allowed_when_stream_registered(self):
        feed._feed_live = True
        feed._stream_writer = object()
        self.assertIsNone(feed.trendbar_fetch_blocked_reason())

    def test_block_clears_after_backoff(self):
        feed._note_trendbar_block("test error", retry_s=0.01)
        self.assertFalse(feed._trendbar_fetch_allowed())
        time.sleep(0.02)
        feed._feed_live = False
        self.assertTrue(feed._trendbar_fetch_allowed())

    def test_apply_live_tick_updates_forming_bar(self):
        feed._spot_cache["XAUUSD"] = (2650.0, 2652.0, time.monotonic())
        step = feed._TF_MINUTES["15m"] * 60_000
        now_ms = (int(time.time() * 1000) // step) * step
        rows = [[now_ms - step, 1, 2, 0.5, 1.5, 0]]
        out = feed._apply_live_tick_to_rows(rows, "XAUUSD", "15m", 10)
        self.assertGreater(float(out[-1][4]), 2600.0)


if __name__ == "__main__":
    unittest.main()
