"""cTrader feed auth helpers."""
import unittest
from unittest.mock import MagicMock

from app.services import ctrader_price_feed as feed


class TestCtraderFeedAuth(unittest.TestCase):
    def test_is_live_for_ctid_reads_json(self):
        prefs = MagicMock(
            ctrader_accounts='[{"ctidTraderAccountId":999,"isLive":false}]',
        )
        self.assertFalse(feed._is_live_for_ctid(prefs, 999))
        self.assertTrue(feed._is_live_for_ctid(prefs, 111))


if __name__ == "__main__":
    unittest.main()
