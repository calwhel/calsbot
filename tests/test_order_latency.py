"""Live order latency instrumentation + stale-signal guard."""
import time
import unittest
from unittest.mock import patch

from app.services.order_latency import OrderLatency, new_order_latency
from app.services.order_stale_guard import check_signal_stale


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
    def test_aborts_old_signal(self):
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_mono=time.monotonic() - 60.0,
            max_age_s=30.0,
        )
        self.assertIsNotNone(stale)
        self.assertIn("signal stale", stale[0])

    @patch("app.services.order_stale_guard._live_mid", return_value=2665.0)
    def test_aborts_excessive_slippage(self, _mid):
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_mono=time.monotonic() - 2.0,
            max_age_s=30.0,
        )
        self.assertIsNotNone(stale)
        self.assertIn("price moved", stale[0])

    @patch("app.services.order_stale_guard._live_mid", return_value=2650.5)
    def test_allows_fresh_tight_signal(self, _mid):
        stale = check_signal_stale(
            symbol="XAUUSD",
            direction="LONG",
            signal_price=2650.0,
            signal_mono=time.monotonic() - 2.0,
            max_age_s=30.0,
        )
        self.assertIsNone(stale)


if __name__ == "__main__":
    unittest.main()
