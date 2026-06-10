"""Synthetic metal candles + provider trace when all kline sources miss."""
import unittest
from unittest.mock import AsyncMock, patch

from app.services import tradfi_prices as tp


def _bars(n: int, close: float = 2650.0) -> list:
    return [
        [1_700_000_000_000 + i * 60_000, close, close + 1, close - 1, close, 1.0]
        for i in range(n)
    ]


class TestMetalSyntheticCandles(unittest.IsolatedAsyncioTestCase):
    async def test_build_synthetic_from_spot(self):
        with patch.object(
            tp, "_resolve_metal_spot_price",
            new_callable=AsyncMock,
            return_value=(2650.0, "metals_spot_feed"),
        ):
            rows = await tp.build_synthetic_metal_candles("XAUUSD", "15m", 80)
        self.assertEqual(len(rows), 80)
        self.assertGreater(rows[-1][4], 0)

    async def test_fetch_metal_live_synthetic_when_all_miss(self):
        with patch.object(
            tp, "_fetch_ctrader_klines", new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_binance_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_fmp_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_kraken_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "build_synthetic_metal_candles",
            new_callable=AsyncMock,
            return_value=_bars(80),
        ), patch(
            "app.services.ctrader_price_feed.broker_session_ready", return_value=False,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "15m", 80)
        self.assertEqual(len(rows), 80)
        self.assertEqual(
            tp.get_metal_kline_source("XAUUSD", "15m", 80),
            "synthetic",
        )

    async def test_get_klines_impl_uses_synthetic_not_gc_f(self):
        with patch.object(
            tp, "fetch_metal_live_candles", new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "build_synthetic_metal_candles",
            new_callable=AsyncMock,
            return_value=_bars(80),
        ), patch.object(
            tp, "_fetch_yahoo_chart_klines", new_callable=AsyncMock,
            return_value=_bars(80, 4350.0),
        ) as mock_yahoo:
            rows = await tp._get_klines_impl(
                "XAUUSD", "forex", "15m", 80, for_backtest=False,
            )
        self.assertEqual(len(rows), 80)
        mock_yahoo.assert_not_called()


if __name__ == "__main__":
    unittest.main()
