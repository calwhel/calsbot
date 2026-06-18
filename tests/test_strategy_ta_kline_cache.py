"""Conditions-phase kline cache + fetch budget in strategy_ta."""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio

from app.services.strategy_ta import (
    StrategyEvalCancelled,
    _get_klines,
    conditions_budget_scope,
    evaluate_strategy_conditions,
)


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


class TestSmcColdCacheSharedFetch(unittest.IsolatedAsyncioTestCase):
    async def test_id214_shape_prefetches_once_for_smc_stack(self):
        cfg = {
            "asset_class": "forex",
            "entry_conditions": {
                "operator": "AND",
                "conditions": [
                    {"type": "order_block", "timeframe": "15m", "ob_type": "bullish"},
                    {"type": "fvg", "timeframe": "15m", "condition": "price_in_gap"},
                    {"type": "market_structure", "timeframe": "15m", "condition": "bos_bullish"},
                ],
            },
        }
        price_data = {"price": 2300.0, "_asset_class": "forex"}
        fetch_state = {"count": 0}

        def _bars(n: int = 240):
            out = []
            base = 2300.0
            for i in range(n):
                o = base + (i * 0.05)
                c = o + 0.02
                h = c + 0.03
                l = o - 0.03
                out.append([i * 60_000, o, h, l, c, 100.0 + i])
            return out

        async def _cold_cache_fetch(symbol, interval, limit, http_client, cache):
            key = (symbol, interval, "__smc")
            if cache is not None and key in cache:
                rows = cache[key]
                return rows[-limit:] if len(rows) > limit else rows
            fetch_state["count"] += 1
            await asyncio.sleep(2.5)
            rows = _bars()
            if cache is not None:
                cache[key] = rows
            return rows[-limit:] if len(rows) > limit else rows

        with patch("app.services.strategy_ta._get_klines", side_effect=_cold_cache_fetch):
            t0 = asyncio.get_running_loop().time()
            passed, details = await evaluate_strategy_conditions(
                cfg,
                "XAUUSD",
                price_data,
                {},
                MagicMock(),
            )
            elapsed = asyncio.get_running_loop().time() - t0

        self.assertFalse(passed)
        self.assertEqual(fetch_state["count"], 1)
        self.assertLess(elapsed, 3.4)
        self.assertEqual(len(details), 3)

    async def test_xauusd_metals_chain_cancels_under_conditions_budget(self):
        cfg = {
            "asset_class": "forex",
            "entry_conditions": {
                "operator": "AND",
                "conditions": [
                    {"type": "order_block", "timeframe": "15m", "ob_type": "bullish"},
                ],
            },
        }
        price_data = {"price": 2300.0, "_asset_class": "forex"}
        cancellation_seen = asyncio.Event()

        async def _slow_metal_chain(*_args, **_kwargs):
            try:
                await asyncio.sleep(10.0)
                return []
            except asyncio.CancelledError:
                cancellation_seen.set()
                raise

        with patch(
            "app.services.tradfi_prices.fetch_metal_live_candles",
            side_effect=_slow_metal_chain,
        ):
            with self.assertRaises(StrategyEvalCancelled) as ctx:
                with conditions_budget_scope(0.05):
                    await asyncio.wait_for(
                        evaluate_strategy_conditions(
                            cfg,
                            "XAUUSD",
                            price_data,
                            {},
                            MagicMock(),
                        ),
                        timeout=0.05,
                    )

        self.assertEqual(ctx.exception.reason, "conditions_timeout")
        self.assertTrue(cancellation_seen.is_set())


if __name__ == "__main__":
    unittest.main()
