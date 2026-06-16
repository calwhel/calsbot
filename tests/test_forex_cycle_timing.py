"""Forex executor cycle timing instrumentation and prefetch dedup."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.strategy_executor import (
    ExecutorCycleCtx,
    _log_cycle_timing,
    _prefetch_price_ta_for_cycle,
    _price_ta_cache_key,
    _price_ta_cache_lookup,
)


class TestCycleTimingLog(unittest.TestCase):
    def test_cycle_log_format(self):
        with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
            _log_cycle_timing(
                shard_index=0,
                strategy_count=57,
                prefetch_ms=1200.4,
                eval_ms=45000.2,
                fire_ms=120.0,
                total_ms=46320.6,
                prefetch_stats={"unique_keys": 5, "symbol_refs": 30},
                strategy_timings_ms=[(1, 5000), (2, 3000), (3, 1000)],
            )
        joined = "\n".join(cm.output)
        self.assertIn(
            "[cycle] shard=0 strategies=57 prefetch=1200ms eval=45000ms "
            "fire=120ms total=46321ms prefetch_dedup=5/30",
            joined,
        )
        self.assertIn("[cycle] slowest id=1 eval=5000ms id=2 eval=3000ms id=3 eval=1000ms", joined)


class TestPrefetchDedup(unittest.TestCase):
    def test_dedupes_symbol_timeframe_across_snapshots(self):
        snapshots = [
            {
                "id": 1,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["XAUUSD"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
            {
                "id": 2,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["XAUUSD", "EURUSD"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
            {
                "id": 3,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["XAUUSD"]},
                    "entry_conditions": {"conditions": [{"timeframe": "1h"}]},
                },
            },
        ]

        async def _run():
            http = MagicMock()
            with patch(
                "app.services.strategy_executor._fetch_price_and_ta",
                new_callable=AsyncMock,
                return_value={"price": 1.0},
            ) as mock_fetch:
                stats = await _prefetch_price_ta_for_cycle(
                    snapshots,
                    http,
                    {"forex"},
                    label="Test",
                )
            return stats, mock_fetch.await_count

        stats, fetch_count = asyncio.run(_run())
        self.assertEqual(stats["symbol_refs"], 4)
        self.assertEqual(stats["unique_keys"], 3)
        self.assertEqual(fetch_count, 3)


class TestPriceTaCacheLookup(unittest.TestCase):
    def test_cache_key_includes_asset_class_and_timeframe(self):
        key = _price_ta_cache_key("XAUUSD", "forex", "15m")
        self.assertEqual(key, "forex:XAUUSD:15m")

    def test_lookup_miss_when_empty(self):
        self.assertIsNone(
            _price_ta_cache_lookup("EURUSD", "forex", "15m"),
        )


class TestExecutorCycleCtx(unittest.TestCase):
    def test_shared_kline_cache_field(self):
        ctx = ExecutorCycleCtx(shard_index=1)
        ctx.kline_cache[("XAUUSD", "15m", 200, "forex")] = [1, 2, 3]
        self.assertEqual(len(ctx.kline_cache), 1)
