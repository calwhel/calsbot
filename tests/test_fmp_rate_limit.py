"""FMP rate limiter + stale kline serve during 429 backoff."""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.services import fmp_price_feed as fmp


class TestFmpRateLimit(unittest.IsolatedAsyncioTestCase):
    async def test_get_klines_serves_stale_during_backoff(self):
        fmp._KLINE_CACHE.clear()
        fmp._FMP_BACKOFF_UNTIL = datetime.utcnow() + timedelta(seconds=60)
        rows = [[1_700_000_000_000, 1, 2, 0.5, 1.5, 0]]
        fmp._KLINE_CACHE[("EURUSD", "15min", 80)] = (rows, datetime.utcnow())

        with patch.object(fmp, "_fmp_api_key", return_value="test-key"), patch.object(
            fmp, "_fetch_fmp_chart_once", new_callable=AsyncMock,
        ) as mock_fetch:
            out = await fmp.get_klines("EURUSD", "forex", "15m", 80)
        self.assertEqual(out, rows)
        mock_fetch.assert_not_called()

    async def test_get_klines_single_flight_dedupes(self):
        fmp._KLINE_CACHE.clear()
        fmp._FMP_BACKOFF_UNTIL = None
        fmp._FMP_KLINE_INFLIGHT.clear()
        call_count = 0

        async def _slow_impl(*_a, **_k):
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.05)
            return [[1, 1, 1, 1, 1, 0]]

        with patch.object(fmp, "_get_klines_impl", side_effect=_slow_impl):
            results = await asyncio.gather(
                fmp.get_klines("EURUSD", "forex", "15m", 80),
                fmp.get_klines("EURUSD", "forex", "15m", 80),
            )
        self.assertEqual(call_count, 1)
        self.assertEqual(len(results[0]), 1)
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
