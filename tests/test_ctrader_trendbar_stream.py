"""Trendbar fetch on live spot stream (no second socket)."""
import asyncio
import os
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

    def test_apply_tick_preserves_closed_bars(self):
        step = feed._TF_MINUTES["5m"] * 60_000
        now_ms = (int(time.time() * 1000) // step) * step
        closed = [now_ms - 2 * step, 100.0, 110.0, 90.0, 105.0, 10.0]
        forming = [now_ms - step, 105.0, 108.0, 104.0, 106.0, 5.0]
        rows = [list(closed), list(forming)]
        out = feed._apply_tick_to_bar_rows(rows, 2651.0, now_ms - step + 60_000, "5m", 10)
        self.assertEqual(out[0], closed)
        self.assertEqual(float(out[1][4]), 2651.0)

    def test_synthesize_on_return_remote_uses_postgres_price(self):
        feed._feed_live = False
        step = feed._TF_MINUTES["5m"] * 60_000
        stale_ts = int(time.time() * 1000) - 10 * 60_000
        stale_bar_ts = (stale_ts // step) * step
        rows = [[stale_bar_ts, 1, 2, 0.5, 1.5, 0.0]]
        with mock.patch.object(feed, "ctrader_spot_ready", return_value=True), mock.patch.object(
            feed, "get_price", return_value=2650.0,
        ):
            out = feed._synthesize_klines_on_return(rows, "XAUUSD", "5m", 60)
        self.assertGreater(len(out), 1)
        self.assertLess(feed._newest_bar_age_s(out), 400.0)

    def test_synthesize_on_return_skipped_without_spot(self):
        feed._feed_live = False
        step = feed._TF_MINUTES["5m"] * 60_000
        stale_ts = int(time.time() * 1000) - 10 * 60_000
        stale_bar_ts = (stale_ts // step) * step
        rows = [[stale_bar_ts, 1, 2, 0.5, 1.5, 0.0]]
        with mock.patch.object(feed, "ctrader_spot_ready", return_value=False):
            out = feed._synthesize_klines_on_return(rows, "XAUUSD", "5m", 60)
        self.assertEqual(out, rows)

    def test_fetch_trendbars_blocked_on_portal_worker(self):
        feed._feed_live = False
        feed._stream_writer = None
        with mock.patch.dict(
            os.environ,
            {"EXECUTOR_STANDALONE": "", "DISABLE_EXECUTOR_IN_GUNICORN": "1"},
            clear=False,
        ):
            with mock.patch.object(feed, "_standalone_trendbar_fetch_allowed", return_value=False):
                out = asyncio.get_event_loop().run_until_complete(
                    feed._fetch_trendbars("XAUUSD", "5m", 60, "tok", 1),
                )
        self.assertEqual(out, [])

    def test_synthesize_on_return_feed_worker_idempotent(self):
        feed._feed_live = True
        step = feed._TF_MINUTES["5m"] * 60_000
        now_ms = (int(time.time() * 1000) // step) * step
        rows = [[now_ms, 2650.0, 2652.0, 2648.0, 2651.0, 0.0]]
        with mock.patch.object(feed, "ctrader_spot_ready", return_value=True), mock.patch.object(
            feed, "get_price", return_value=2651.0,
        ):
            out = feed._synthesize_klines_on_return(
                rows, "XAUUSD", "5m", 60, log_remote=False,
            )
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out[0][4]), 2651.0)
        self.assertEqual(int(out[0][0]), now_ms)


if __name__ == "__main__":
    unittest.main()
