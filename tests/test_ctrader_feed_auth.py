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
        feed._last_auth_error = None
        feed._auth_backoff_until = 0.0

    def test_is_live_for_ctid_reads_json(self):
        prefs = MagicMock(
            ctrader_accounts='[{"ctidTraderAccountId":999,"isLive":false}]',
        )
        self.assertFalse(feed._is_live_for_ctid(prefs, 999))
        self.assertTrue(feed._is_live_for_ctid(prefs, 111))

    def test_broker_session_ready_false_when_feed_live(self):
        """Spot feed owns the socket — trendbars must not open a second session."""
        feed._feed_live = True
        self.assertFalse(feed.broker_session_ready())

    def test_broker_session_ready_false_on_stale_spot_only(self):
        feed._spot_cache["XAUUSD"] = (2650.0, 2651.0, time.monotonic())
        self.assertFalse(feed.broker_session_ready("XAUUSD"))

    def test_broker_session_ready_when_stream_creds_idle(self):
        feed._stream_creds = ("tok", 47516246, 1, feed._HOST_DEMO)
        self.assertTrue(feed.broker_session_ready("XAUUSD"))

    def test_broker_session_ready_false_on_terminal_auth(self):
        feed._last_auth_error = "CH_ACCESS_TOKEN_INVALID: Invalid access token"
        feed._spot_cache["XAUUSD"] = (2650.0, 2651.0, time.monotonic())
        self.assertFalse(feed.broker_session_ready("XAUUSD"))

    def test_get_stream_creds_roundtrip(self):
        feed._stream_creds = ("tok", 47516246, 1, feed._HOST_DEMO)
        self.assertEqual(feed.get_stream_creds()[1], 47516246)

    def test_trendbar_allowed_on_live_stream_when_registered(self):
        feed._feed_live = True
        feed._stream_writer = object()
        feed._trendbar_block_until = 0.0
        feed._trendbar_block_reason = None
        self.assertTrue(feed._trendbar_fetch_allowed())

    def test_trendbar_blocked_when_live_without_stream(self):
        feed._feed_live = True
        feed._stream_writer = None
        feed._trendbar_block_until = 0.0
        reason = feed.trendbar_fetch_blocked_reason()
        self.assertIn("stream session", reason or "")


if __name__ == "__main__":
    unittest.main()
