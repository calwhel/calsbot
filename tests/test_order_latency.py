"""Live order latency instrumentation + stale-signal guard."""
import time
import unittest
from unittest.mock import patch

from app.services.order_latency import OrderLatency, new_order_latency
from app.services.order_stale_guard import (
    check_signal_stale,
    get_stale_verdict,
    set_stale_verdict,
)


class TestOrderLatency(unittest.TestCase):
    def test_summary_log_format(self):
        lat = OrderLatency(execution_id=42, signal_mono=100.0)
        lat.queued_mono = 100.1
        lat.dequeue_mono = 100.2
        lat.submitted_mono = 100.25
        lat.broker_ack_mono = 100.4
        lat.fill_mono = 100.55
        with self.assertLogs("app.services.order_latency", level="INFO") as logs:
            lat.log_summary(outcome="fill")
        msg = logs.output[-1]
        self.assertIn("[order-latency] exec=42", msg)
        self.assertIn("signal→queued=", msg)
        self.assertIn("total=", msg)
        self.assertIn("outcome=fill", msg)

    def test_new_order_latency_factory(self):
        lat = new_order_latency(7)
        self.assertEqual(lat.execution_id, 7)
        self.assertGreater(lat.signal_mono, 0)


class TestStaleSignalGuard(unittest.TestCase):
    def setUp(self):
        from app.services import order_stale_guard as osg
        osg._STALE_VERDICT.clear()

    def test_aborts_old_signal(self):
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_generated_at=time.time() - 60.0,
            max_age_s=30.0,
            price_source="spot_live",
            live_source="ctrader",
            execution_id=101,
        )
        self.assertIsNotNone(stale)
        self.assertIn("signal stale", stale[0])
        self.assertEqual(get_stale_verdict(101), "blocked")

    @patch("app.services.order_stale_guard._current_price_same_source")
    def test_aborts_excessive_slippage_same_source(self, mock_now):
        mock_now.return_value = (2665.0, "ctrader")
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_generated_at=time.time() - 10.0,
            max_age_s=30.0,
            price_source="spot_live",
            live_source="ctrader",
            execution_id=102,
        )
        self.assertIsNotNone(stale)
        self.assertIn("price moved", stale[0])
        self.assertIn("sig_src=spot:ctrader", stale[0])

    @patch("app.services.order_stale_guard._current_price_same_source")
    def test_allows_fresh_tight_signal(self, mock_now):
        mock_now.return_value = (2650.5, "ctrader")
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_generated_at=time.time() - 2.0,
            max_age_s=30.0,
            price_source="spot_live",
            live_source="ctrader",
            execution_id=103,
        )
        self.assertIsNone(stale)
        self.assertEqual(get_stale_verdict(103), "allowed")

    @patch("app.services.order_stale_guard._current_price_same_source")
    def test_does_not_block_cross_source_drift(self, mock_now):
        mock_now.return_value = (2720.0, "yahoo_gc")
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_generated_at=time.time() - 1.0,
            price_source="kline_close",
            kline_source="binance",
            execution_id=104,
        )
        self.assertIsNone(stale)
        self.assertEqual(get_stale_verdict(104), "allowed")

    @patch("app.services.order_stale_guard._current_price_same_source")
    def test_implausible_drift_not_blocked(self, mock_now):
        mock_now.return_value = (2720.0, "ctrader")
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_generated_at=time.time() - 1.0,
            max_age_s=30.0,
            price_source="spot_live",
            live_source="ctrader",
            execution_id=105,
        )
        self.assertIsNone(stale)
        self.assertEqual(get_stale_verdict(105), "allowed")

    def test_once_per_signal_verdict_cached(self):
        set_stale_verdict(200, "blocked", "price moved 20 pips")
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_generated_at=time.time() - 1.0,
            execution_id=200,
        )
        self.assertIsNotNone(stale)
        self.assertIn("price moved 20 pips", stale[0])


if __name__ == "__main__":
    unittest.main()
