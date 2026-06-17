"""Conditions-phase kline cache + fetch budget in strategy_ta."""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio

from app.services.strategy_ta import _get_klines


class TestTradfiKlineCache(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_larger_limit_fetch_for_smaller_request(self):
        cache = {"__asset_class__": "forex"}
        bars = [[i, 1, 1, 1, 1, 1] for i in range(200)]

        async def _fake_klines(*_a, **_k):
            return bars

        with patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            side_effect=_fake_klines,
        ) as mock_fetch:
            first = await _get_klines("EURUSD", "15m", 200, MagicMock(), cache)
            second = await _get_klines("EURUSD", "15m", 80, MagicMock(), cache)

        self.assertEqual(len(first), 200)
        self.assertEqual(len(second), 80)
        self.assertEqual(mock_fetch.await_count, 1)

    async def test_tradfi_fetch_times_out_under_prefetch_fast(self):
        cache = {"__asset_class__": "forex"}

        async def _slow_klines(*_a, **_k):
            await asyncio.sleep(0.2)
            return []

        with patch(
            "app.services.tradfi_prices.get_klines",
            new_callable=AsyncMock,
            side_effect=_slow_klines,
        ), patch(
            "app.services.prefetch_fast.prefetch_fast_active",
            return_value=True,
        ), patch(
            "app.services.prefetch_fast.SYMBOL_BUDGET_S",
            0.05,
        ):
            out = await _get_klines("EURUSD", "15m", 200, MagicMock(), cache)

        self.assertIsNone(out)


if __name__ == "__main__":
    unittest.main()
