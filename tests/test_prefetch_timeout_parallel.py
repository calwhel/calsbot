"""Prefetch per-fetch logs, 2.5s budget, concurrent gather, cache fallback."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import logging
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import tradfi_prices as tp
from app.services.prefetch_fast import SYMBOL_BUDGET_S
from app.services.strategy_executor import (
    _PRICE_TA_CACHE,
    _PRICE_TA_INFLIGHT,
    _fetch_price_ta_singleflight,
    _prefetch_fallback_price_ta,
    _prefetch_price_ta_for_cycle,
    _price_ta_from_klines,
)


class TestPrefetchBudget(unittest.TestCase):
    def test_symbol_budget_is_sub_three_seconds(self):
        self.assertLessEqual(SYMBOL_BUDGET_S, 2.0)


class TestPeekCachedKlines(unittest.TestCase):
    def test_prefers_ctrader_cache(self):
        rows = [[1, 1, 2, 0.5, 1.5, 0], [2, 1.5, 2, 1, 1.8, 0]]
        try:
            from app.services import ctrader_price_feed as ctf
            ctf._kline_cache[("EURUSD", "15m", 200)] = (rows, 1.0)
            got, src = tp.peek_cached_klines("EURUSD", "forex", "15m", 200)
            self.assertEqual(len(got), 2)
            self.assertEqual(src, "ctrader-cache")
        finally:
            from app.services import ctrader_price_feed as ctf
            ctf._kline_cache.pop(("EURUSD", "15m", 200), None)


class TestPrefetchFallback(unittest.TestCase):
    def tearDown(self):
        _PRICE_TA_CACHE.clear()

    def test_price_ta_from_klines_seeds_cache(self):
        kl = [
            [1000, 1.0, 1.1, 0.9, 1.0, 0],
            [2000, 1.0, 1.1, 0.9, 1.05, 0],
            [3000, 1.05, 1.15, 1.0, 1.1, 0],
        ]
        result = _price_ta_from_klines("EURUSD", "forex", "15m", kl, kline_source="ctrader-cache")
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["price"], 1.05)
        key = "forex:EURUSD:15m"
        self.assertIn(key, _PRICE_TA_CACHE)

    def test_stale_price_ta_used_on_timeout(self):
        stale = {"price": 1.234, "kline_source": "ctrader"}
        _PRICE_TA_CACHE["forex:GBPUSD:15m"] = (stale, datetime.utcnow() - timedelta(hours=1))
        got, src = _prefetch_fallback_price_ta("GBPUSD", "forex", "15m")
        self.assertEqual(got, stale)
        self.assertEqual(src, "price_ta_cache")
        _PRICE_TA_CACHE.pop("forex:GBPUSD:15m", None)


class TestPrefetchConcurrent(unittest.TestCase):
    def tearDown(self):
        _PRICE_TA_CACHE.clear()
        _PRICE_TA_INFLIGHT.clear()

    def test_slow_symbol_does_not_block_fast_symbol(self):
        snapshots = [
            {
                "id": 1,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["SLOWPAIR"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
            {
                "id": 2,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["FASTPAIR"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
        ]

        async def _fake_fetch(sym, http_client, ac, **kwargs):
            if sym == "SLOWPAIR":
                await asyncio.sleep(0.4)
                return {"price": 1.0, "kline_source": "slow"}
            return {"price": 2.0, "kline_source": "fast"}

        async def _run():
            http = MagicMock()
            with patch(
                "app.services.strategy_executor._fetch_price_and_ta",
                side_effect=_fake_fetch,
            ):
                with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
                    stats = await _prefetch_price_ta_for_cycle(
                        snapshots, http, {"forex"}, label="Test",
                    )
            return stats, "\n".join(cm.output)

        stats, logs = asyncio.run(_run())
        self.assertEqual(stats["fetched"], 2)
        self.assertIn("sym=SLOWPAIR", logs)
        self.assertIn("sym=FASTPAIR", logs)
        self.assertIn("executor=Test", logs)
        self.assertIn("provider=fast", logs)
        self.assertIn("provider=slow", logs)

    def test_timeout_uses_cached_fallback(self):
        snapshots = [
            {
                "id": 1,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["XAUUSD"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
        ]
        kl = [[i * 1000, 1, 2, 0.5, 1.0 + i * 0.01, 0] for i in range(20)]

        async def _slow_fetch(*args, **kwargs):
            await asyncio.sleep(10)
            return {"price": 99}

        async def _run():
            http = MagicMock()
            with patch(
                "app.services.strategy_executor._fetch_price_and_ta",
                side_effect=_slow_fetch,
            ):
                with patch(
                    "app.services.prefetch_fast.SYMBOL_BUDGET_S",
                    0.05,
                ):
                    with patch(
                        "app.services.strategy_executor._prefetch_fallback_price_ta",
                        return_value=({"price": 1.5, "kline_source": "ctrader-cache"}, "ctrader-cache"),
                    ):
                        with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
                            stats = await _prefetch_price_ta_for_cycle(
                                snapshots, http, {"forex"}, label="Test",
                            )
            return stats, "\n".join(cm.output)

        stats, logs = asyncio.run(_run())
        self.assertEqual(stats["fallback"], 1)
        self.assertIn("result=fallback", logs)


class TestPriceFetchBudgetAndDedupe(unittest.TestCase):
    def tearDown(self):
        _PRICE_TA_CACHE.clear()
        _PRICE_TA_INFLIGHT.clear()

    def test_price_fetch_timeout_is_bounded(self):
        async def _slow_fetch(*_args, **_kwargs):
            await asyncio.sleep(0.3)
            return {"price": 1.2}

        async def _run():
            http = MagicMock()
            with patch(
                "app.services.strategy_executor._fetch_price_and_ta",
                side_effect=_slow_fetch,
            ):
                t0 = asyncio.get_running_loop().time()
                out = await _fetch_price_ta_singleflight(
                    "XAUUSD",
                    http,
                    "forex",
                    timeframe="15m",
                    prefetch=True,
                    budget_s=0.05,
                )
                elapsed = asyncio.get_running_loop().time() - t0
            return out, elapsed

        out, elapsed = asyncio.run(_run())
        self.assertIsNone(out)
        self.assertLess(elapsed, 0.2)

    def test_price_fetch_singleflight_dedupes_concurrent_calls(self):
        calls = {"n": 0}

        async def _slow_fetch(*_args, **_kwargs):
            calls["n"] += 1
            await asyncio.sleep(0.05)
            return {"price": 2.5, "kline_source": "test"}

        async def _run():
            http = MagicMock()
            with patch(
                "app.services.strategy_executor._fetch_price_and_ta",
                side_effect=_slow_fetch,
            ):
                return await asyncio.gather(*[
                    _fetch_price_ta_singleflight(
                        "XAUUSD",
                        http,
                        "forex",
                        timeframe="15m",
                        prefetch=True,
                        budget_s=0.5,
                    )
                    for _ in range(5)
                ])

        out = asyncio.run(_run())
        self.assertEqual(calls["n"], 1)
        self.assertEqual(len(out), 5)
        self.assertTrue(all(isinstance(item, dict) and item.get("price") == 2.5 for item in out))
        self.assertEqual(_PRICE_TA_INFLIGHT, {})


if __name__ == "__main__":
    unittest.main()
