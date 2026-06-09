"""cTrader feed auth helpers."""
import time
import unittest
from unittest.mock import MagicMock

from app.services import ctrader_price_feed as feed


class TestCtraderFeedAuth(unittest.TestCase):
    def setUp(self):
        feed._feed_live = False
        feed._stream_creds = None
        feed._spot_cache.clear()

    def test_is_live_for_ctid_reads_json(self):
        prefs = MagicMock(
            ctrader_accounts='[{"ctidTraderAccountId":999,"isLive":false}]',
        )
        self.assertFalse(feed._is_live_for_ctid(prefs, 999))
        self.assertTrue(feed._is_live_for_ctid(prefs, 111))

    def test_broker_session_ready_when_live(self):
        feed._feed_live = True
        self.assertTrue(feed.broker_session_ready())

    def test_broker_session_ready_when_fresh_spot(self):
        feed._spot_cache["XAUUSD"] = (2650.0, 2651.0, time.monotonic())
        self.assertTrue(feed.broker_session_ready("XAUUSD"))

    def test_get_stream_creds_roundtrip(self):
        feed._stream_creds = ("tok", 47516246, 1, feed._HOST_DEMO)
        self.assertEqual(feed.get_stream_creds()[1], 47516246)


if __name__ == "__main__":
    unittest.main()
