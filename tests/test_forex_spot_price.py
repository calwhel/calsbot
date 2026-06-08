"""Unified forex / metals spot quote resolver."""
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch


class ForexSpotPriceTest(unittest.IsolatedAsyncioTestCase):
    async def test_ctrader_first(self):
        from app.services import forex_spot_price as fsp

        with patch("app.services.ctrader_price_feed.get_price", return_value=2650.5), patch(
            "app.services.ctrader_price_feed.get_bid_ask", return_value=(2650.4, 2650.6)
        ):
            q = await fsp.get_forex_spot_quote("XAUUSD")
        self.assertIsNotNone(q)
        self.assertEqual(q["source"], "ctrader")
        self.assertAlmostEqual(q["mid"], 2650.5)

    async def test_shared_store_when_ctrader_cold(self):
        from app.services import forex_spot_price as fsp

        row = {
            "mid": 4318.2,
            "bid": None,
            "ask": None,
            "source": "coinbase",
            "updated_at": datetime.utcnow(),
        }
        with patch("app.services.ctrader_price_feed.get_price", return_value=None), patch(
            "app.services.spot_price_store.get_tick", return_value=row
        ):
            q = await fsp.get_forex_spot_quote("XAUUSD")
        self.assertEqual(q["source"], "coinbase")
        self.assertAlmostEqual(q["mid"], 4318.2)

    async def test_metals_fetch_on_demand(self):
        from app.services import forex_spot_price as fsp

        with patch("app.services.ctrader_price_feed.get_price", return_value=None), patch(
            "app.services.spot_price_store.get_tick", side_effect=[None, {
                "mid": 2651.0,
                "bid": None,
                "ask": None,
                "source": "binance",
                "updated_at": datetime.utcnow(),
            }]
        ), patch("app.services.metals_spot_feed.fetch_now", new=AsyncMock(return_value=2651.0)):
            q = await fsp.get_forex_spot_quote("XAUUSD")
        self.assertEqual(q["source"], "binance")


if __name__ == "__main__":
    unittest.main()
