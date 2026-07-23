"""Inline tick→bar updates in cTrader spot callback."""
import time
import unittest
from unittest.mock import patch

from app.services import ctrader_price_feed as feed


class TestInlineTickBars(unittest.TestCase):
    def setUp(self):
        feed._kline_cache.clear()
        feed._last_kline_update.clear()
        feed._last_peer_persist.clear()
        feed._spot_cache.clear()

    def test_forming_bar_ohlc_tracks_ticks(self):
        step = feed._TF_MINUTES["15m"] * 60_000
        bar_ts = (int(time.time() * 1000) // step) * step
        rows = [[bar_ts, 1.0, 1.1, 0.9, 1.0, 0.0]]
        out = feed._apply_tick_to_bar_rows(rows, 1.05, bar_ts + 1000, "15m", 10)
        self.assertEqual(int(out[-1][0]), bar_ts)
        self.assertEqual(out[-1][4], 1.05)
        self.assertEqual(out[-1][2], 1.1)
        self.assertEqual(out[-1][3], 0.9)

        out2 = feed._apply_tick_to_bar_rows(out, 0.85, bar_ts + 2000, "15m", 10)
        self.assertEqual(out2[-1][4], 0.85)
        self.assertEqual(out2[-1][3], 0.85)
        self.assertEqual(out2[-1][2], 1.1)

        out3 = feed._apply_tick_to_bar_rows(out2, 1.2, bar_ts + 3000, "15m", 10)
        self.assertEqual(out3[-1][4], 1.2)
        self.assertEqual(out3[-1][2], 1.2)

    def test_bar_rolls_at_timeframe_boundary(self):
        step = feed._TF_MINUTES["5m"] * 60_000
        bar_ts = (int(time.time() * 1000) // step) * step
        rows = [[bar_ts, 1.0, 1.0, 1.0, 1.0, 0.0]]
        next_bar = bar_ts + step
        out = feed._apply_tick_to_bar_rows(rows, 1.5, next_bar + 500, "5m", 10)
        self.assertEqual(len(out), 2)
        self.assertEqual(int(out[-2][0]), bar_ts)
        self.assertEqual(int(out[-1][0]), next_bar)
        self.assertEqual(out[-1][1], 1.5)
        self.assertEqual(out[-1][4], 1.5)

    def test_bar_open_ts_aligns_to_boundary(self):
        step = feed._TF_MINUTES["15m"] * 60_000
        ts = step * 7 + 123_456
        self.assertEqual(feed._bar_open_ts_ms(ts, "15m"), step * 7)

    def test_update_kline_cache_on_tick_all_timeframes(self):
        step15 = feed._TF_MINUTES["15m"] * 60_000
        step5 = feed._TF_MINUTES["5m"] * 60_000
        now_ms = int(time.time() * 1000)
        bar15 = (now_ms // step15) * step15
        bar5 = (now_ms // step5) * step5
        feed._kline_cache[("EURUSD", "15m", 80)] = (
            [[bar15, 1.08, 1.09, 1.07, 1.08, 0.0]],
            0.0,
        )
        feed._kline_cache[("EURUSD", "5m", 80)] = (
            [[bar5, 1.08, 1.09, 1.07, 1.08, 0.0]],
            0.0,
        )
        feed._update_kline_cache_on_tick("EURUSD", 1.10, now_ms + 1000)
        r15 = feed._kline_cache[("EURUSD", "15m", 80)][0][-1]
        r5 = feed._kline_cache[("EURUSD", "5m", 80)][0][-1]
        self.assertEqual(r15[4], 1.10)
        self.assertEqual(r5[4], 1.10)
        self.assertIn("EURUSD", feed._last_kline_update)

    def test_tick_update_persists_snapshot_for_peers_throttled(self):
        """Tick-rolled bars are persisted to the shared snapshot (once per
        interval per timeframe) so non-feed peers stay fresh during trendbar
        blocks, then throttled on rapid follow-up ticks."""
        step5 = feed._TF_MINUTES["5m"] * 60_000
        now_ms = int(time.time() * 1000)
        bar5 = (now_ms // step5) * step5
        feed._kline_cache[("XAUUSD", "5m", 80)] = (
            [[bar5, 4000.0, 4001.0, 3999.0, 4000.0, 0.0]],
            0.0,
        )
        with patch.object(feed, "_persist_klines_for_peers") as persist, patch.object(
            feed, "_peer_persist_interval_s", return_value=20.0
        ):
            feed._update_kline_cache_on_tick("XAUUSD", 4002.0, now_ms + 1000)
            self.assertEqual(persist.call_count, 1)
            self.assertEqual(persist.call_args[0][0], "XAUUSD")
            self.assertEqual(persist.call_args[0][1], "5m")
            # Immediate follow-up tick is throttled (same interval window).
            feed._update_kline_cache_on_tick("XAUUSD", 4003.0, now_ms + 2000)
            self.assertEqual(persist.call_count, 1)

    def test_stale_rebuild_log_rate_limited(self):
        feed._stale_rebuild_log_at.clear()
        with self.assertLogs("app.services.ctrader_price_feed", level="DEBUG") as cm:
            feed._log_stale_kline_rebuild("XAUUSD", "drift=1.2%", "15m")
            feed._log_stale_kline_rebuild("XAUUSD", "drift=1.5%", "15m")
        warnings = [m for m in cm.output if "WARNING" in m]
        debugs = [m for m in cm.output if "DEBUG" in m]
        self.assertEqual(len(warnings), 1)
        self.assertEqual(len(debugs), 1)


if __name__ == "__main__":
    unittest.main()
