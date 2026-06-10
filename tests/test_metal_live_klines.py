"""Parallel spot-metal kline resolver — no GC=F on live path."""
import unittest
from unittest.mock import AsyncMock, patch

from app.services import tradfi_prices as tp


def _bars(n: int, close: float = 2650.0) -> list:
    return [
        [1_700_000_000_000 + i * 60_000, close, close + 1, close - 1, close, 1.0]
        for i in range(n)
    ]


class TestMetalLiveKlines(unittest.IsolatedAsyncioTestCase):
    async def test_picks_binance_over_kraken(self):
        with patch.object(
            tp, "_fetch_binance_metals_klines",
            new_callable=AsyncMock, return_value=_bars(80),
        ), patch.object(
            tp, "_fetch_kraken_metals_klines",
            new_callable=AsyncMock, return_value=_bars(80, 2649.0),
        ), patch.object(
            tp, "_fetch_fmp_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch(
            "app.services.ctrader_price_feed.is_live", return_value=False,
        ), patch(
            "app.services.ctrader_price_feed.broker_session_ready", return_value=False,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "15m", 80)
        self.assertEqual(len(rows), 80)
        self.assertAlmostEqual(rows[-1][4], 2650.0)

    async def test_ctrader_first_when_feed_live(self):
        with patch.object(
            tp, "_fetch_ctrader_klines",
            new_callable=AsyncMock, return_value=_bars(80),
        ) as mock_ct, patch.object(
            tp, "_fetch_binance_metals_klines",
            new_callable=AsyncMock,
            side_effect=AssertionError("binance should be skipped when cTrader LIVE"),
        ), patch(
            "app.services.ctrader_price_feed.is_live", return_value=True,
        ), patch(
            "app.services.ctrader_price_feed.broker_session_ready", return_value=True,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "5m", 80)
        self.assertEqual(len(rows), 80)
        self.assertGreaterEqual(mock_ct.await_count, 1)

    async def test_ctrader_first_when_broker_session_ready(self):
        with patch.object(
            tp, "_fetch_ctrader_klines",
            new_callable=AsyncMock, return_value=_bars(80),
        ), patch(
            "app.services.ctrader_price_feed.is_live", return_value=False,
        ), patch(
            "app.services.ctrader_price_feed.broker_session_ready", return_value=True,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "5m", 80)
        self.assertEqual(len(rows), 80)

    async def test_ctrader_live_falls_through_on_miss(self):
        """When cTrader misses, externals (including Binance) are still tried."""
        with patch.object(
            tp, "_fetch_ctrader_klines",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_binance_metals_klines",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_kraken_metals_klines",
            new_callable=AsyncMock, return_value=_bars(80),
        ), patch.object(
            tp, "_fetch_fmp_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch(
            "app.services.ctrader_price_feed.is_live", return_value=True,
        ), patch(
            "app.services.ctrader_price_feed.broker_session_ready", return_value=True,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "5m", 80)
        self.assertEqual(len(rows), 80)

    async def test_kraken_fallback_when_binance_empty(self):
        with patch.object(
            tp, "_fetch_binance_metals_klines",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_kraken_metals_klines",
            new_callable=AsyncMock, return_value=_bars(80, 4328.0),
        ), patch.object(
            tp, "_fetch_fmp_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch(
            "app.services.ctrader_price_feed.is_live", return_value=False,
        ), patch(
            "app.services.ctrader_price_feed.broker_session_ready", return_value=False,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "5m", 80)
        self.assertEqual(len(rows), 80)
        self.assertAlmostEqual(rows[-1][4], 4328.0)

    async def test_live_impl_never_returns_gc_f(self):
        with patch.object(
            tp, "fetch_metal_live_candles", new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "build_synthetic_metal_candles",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_yahoo_chart_klines", new_callable=AsyncMock,
            return_value=_bars(80, 4350.0),
        ) as mock_yahoo:
            rows = await tp._get_klines_impl(
                "XAUUSD", "forex", "15m", 80, for_backtest=False,
            )
        self.assertEqual(rows, [])
        mock_yahoo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
