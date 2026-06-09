"""Index executor klines — Yahoo chart scan chain."""
import unittest
from unittest.mock import AsyncMock, patch

from app.services import tradfi_prices as tp


def _bars(n: int) -> list:
    return [[i * 60_000, 1.0, 1.1, 0.9, 1.05, 0.0] for i in range(n)]


class TestIndexScanKlines(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_index_scan_returns_yahoo_at_executor_limit(self):
        yahoo_rows = _bars(80)
        with patch(
            "app.services.ctrader_price_feed.broker_session_ready",
            return_value=False,
        ), patch.object(
            tp, "_fetch_ctrader_klines", new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_yahoo_chart_klines",
            new_callable=AsyncMock,
            return_value=yahoo_rows,
        ) as mock_yf:
            rows = await tp.fetch_index_scan_candles("NAS100", "15m", 80)
        self.assertEqual(len(rows), 80)
        mock_yf.assert_awaited_once()

    async def test_get_klines_live_index_uses_scan_chain(self):
        yahoo_rows = _bars(80)
        with patch.object(
            tp, "fetch_index_scan_candles",
            new_callable=AsyncMock,
            return_value=yahoo_rows,
        ) as mock_scan:
            rows = await tp.get_klines("NAS100", "index", "15m", 80)
        self.assertEqual(len(rows), 80)
        mock_scan.assert_awaited_once()

    async def test_scan_min_bars_accepts_80_for_executor(self):
        self.assertEqual(tp._scan_min_bars(80), 80)
        self.assertLessEqual(tp._scan_min_bars(80), 80)


if __name__ == "__main__":
    unittest.main()
