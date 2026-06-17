"""Paper gold eval when live spot tick is missing."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import strategy_executor as se


def _klines(n: int = 80, close: float = 2650.0) -> list:
    return [
        [1_700_000_000_000 + i * 60_000, close, close + 1, close - 1, close, 1.0]
        for i in range(n)
    ]


class TestMetalPaperEval(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        se._PRICE_TA_CACHE.clear()

    async def test_paper_uses_kline_close_without_live_spot(self):
        with patch.object(
            se, "EXECUTOR_KLINE_BARS", 80,
        ), patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            return_value=_klines(),
        ), patch(
            "app.services.tradfi_prices.get_price_fresh",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.tradfi_prices.get_metal_kline_source",
            return_value="kraken",
        ), patch(
            "app.services.ctrader_price_feed.ctrader_spot_ready",
            return_value=False,
        ), patch(
            "app.services.ctrader_price_feed.get_bid_ask",
            return_value=None,
        ), patch(
            "app.services.tradfi_prices.get_metal_live_for_source",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.metals_spot_feed.get_price",
            return_value=None,
        ), patch(
            "app.services.metals_spot_feed.fetch_now",
            new_callable=AsyncMock,
            return_value=None,
        ):
            out = await se._fetch_price_and_ta(
                "XAUUSD", MagicMock(), "forex",
                timeframe="15m", metal_paper_ok=True,
            )
        self.assertIsNotNone(out)
        self.assertEqual(out["price_source"], "kline_close_paper")
        self.assertAlmostEqual(out["price"], 2650.0)

    async def test_live_uses_metals_spot_feed_when_ctrader_tick_missing(self):
        with patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            return_value=_klines(),
        ), patch(
            "app.services.tradfi_prices.get_price_fresh",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.tradfi_prices.get_metal_kline_source",
            return_value="kraken",
        ), patch(
            "app.services.ctrader_price_feed.ctrader_spot_ready",
            return_value=False,
        ), patch(
            "app.services.ctrader_price_feed.get_bid_ask",
            return_value=None,
        ), patch(
            "app.services.tradfi_prices.get_metal_live_for_source",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.metals_spot_feed.get_price",
            return_value=2651.0,
        ):
            out = await se._fetch_price_and_ta(
                "XAUUSD", MagicMock(), "forex",
                timeframe="15m", metal_paper_ok=False,
            )
        self.assertIsNotNone(out)
        self.assertEqual(out["price_source"], "spot_live")
        self.assertAlmostEqual(out["price"], 2651.0)


if __name__ == "__main__":
    unittest.main()
