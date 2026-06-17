"""cTrader deal close parsing — position_id match + closePositionDetail."""
import os
import unittest
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

from app.services.ctrader_client import _parse_close_from_deals


class TestParseCloseFromDeals(unittest.TestCase):
    def test_picks_close_deal_by_position_id(self):
        open_deal = MagicMock()
        open_deal.positionId = 144726466
        open_deal.HasField = lambda f: f == "executionPrice" and True or False
        open_deal.executionPrice = 1.1000
        open_deal.executionTimestamp = 1000
        open_deal.closePositionDetail = None

        close_deal = MagicMock()
        close_deal.positionId = 144726466
        close_deal.dealId = 99

        def _has(field):
            return field in ("closePositionDetail", "executionTimestamp", "grossProfit")

        close_deal.HasField = _has
        close_deal.executionTimestamp = 2000
        close_detail = MagicMock()
        close_detail.entryPrice = 1.0950
        close_detail.grossProfit = -50
        close_deal.closePositionDetail = close_detail

        parsed = _parse_close_from_deals(
            [open_deal, close_deal],
            144726466,
            entry_hint=1.10,
            direction="LONG",
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed["outcome"], "LOSS")
        self.assertAlmostEqual(parsed["exit_price"], 1.0950, places=4)


if __name__ == "__main__":
    unittest.main()
