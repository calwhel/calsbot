"""Tests for close dedupe, kline staleness, CoinGecko limiter, FMP empty cache."""
import time
import unittest
from datetime import datetime, timedelta
from unittest import mock

from app.services import coingecko_safe as cg
from app.services import fmp_price_feed as fmp
from app.services.close_notify_dedupe import claim_close_notification
from app.services.kline_staleness import (
    cached_klines_stale,
    newest_bar_age_s,
    stale_limit_s,
)


class TestCloseNotifyDedupe(unittest.TestCase):
    def test_claim_sql_shape(self):
        import inspect
        src = inspect.getsource(claim_close_notification)
        self.assertIn("notified_close_at IS NULL", src)
        self.assertIn("RETURNING id", src)

    def test_claim_returns_true_once(self):
        class _Result:
            def __init__(self, row):
                self._row = row

            def fetchone(self):
                return self._row

        db = mock.MagicMock()
        db.execute.side_effect = [
            _Result((42,)),
            _Result(None),
        ]
        self.assertTrue(claim_close_notification(db, 42))
        self.assertFalse(claim_close_notification(db, 42))


class TestKlineStaleness(unittest.TestCase):
    def test_source_agnostic_drift(self):
        now_ms = int(time.time() * 1000)
        rows = [[now_ms, 4088.0, 4088.0, 4088.0, 4088.82, 0.0]]
        stale, detail = cached_klines_stale(
            "XAUUSD",
            rows,
            "15m",
            cache_fetched_at=datetime.utcnow(),
            live_px=4147.10,
            live_updating=True,
        )
        self.assertTrue(stale)
        self.assertIn("drift=", detail)

    def test_stale_limit_is_2x_timeframe(self):
        self.assertEqual(stale_limit_s("15m"), 30 * 60.0)


class TestCoinGeckoLimiter(unittest.TestCase):
    def setUp(self):
        cg._req_times.clear()
        cg._paused_until = 0.0
        cg._last_pause_log = 0.0

    def test_pause_after_budget(self):
        cg._MAX_PER_MIN = 2
        self.assertTrue(cg._record_request())
        self.assertTrue(cg._record_request())
        self.assertFalse(cg._record_request())
        self.assertTrue(cg.coingecko_paused())


class TestFmpEmptyCache(unittest.TestCase):
    def setUp(self):
        fmp._FMP_EMPTY_KLINE_UNTIL.clear()
        fmp._FMP_EMPTY_STRIKES.clear()

    def test_negative_cache_after_two_empties(self):
        fmp._record_fmp_empty_klines("NZDJPY", "1h")
        self.assertFalse(fmp.fmp_klines_marked_unavailable("NZDJPY", "1h"))
        fmp._record_fmp_empty_klines("NZDJPY", "1h")
        self.assertTrue(fmp.fmp_klines_marked_unavailable("NZDJPY", "1h"))

    def test_clear_on_success(self):
        fmp._record_fmp_empty_klines("NZDJPY", "1h")
        fmp._record_fmp_empty_klines("NZDJPY", "1h")
        fmp._clear_fmp_empty_klines("NZDJPY", "1h")
        self.assertFalse(fmp.fmp_klines_marked_unavailable("NZDJPY", "1h"))


if __name__ == "__main__":
    unittest.main()
