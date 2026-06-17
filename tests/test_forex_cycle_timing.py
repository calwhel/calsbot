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
    EvalDiag,
    _PRICE_TA_CACHE,
    _log_cycle_timing,
    _log_eval_diag,
    _prefetch_price_ta_for_cycle,
    _price_ta_cache_key,
    _price_ta_cache_lookup,
    _strategy_cache_asset_class,
)


class TestCycleTimingLog(unittest.TestCase):
    def test_cycle_log_format(self):
        with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
            _log_cycle_timing(
                shard_index=0,
                strategy_count=57,
                setup_ms=800.0,
                prefetch_ms=1200.4,
                eval_ms=45000.2,
                post_ms=200.0,
                fire_ms=120.0,
                total_ms=47320.6,
                prefetch_stats={"unique_keys": 5, "symbol_refs": 30},
                strategy_timings_ms=[(1, 5000), (2, 3000), (3, 1000)],
            )
        joined = "\n".join(cm.output)
        self.assertIn(
            "[cycle] shard=0 strategies=57 setup=800ms prefetch=1200ms eval=45000ms "
            "post=200ms fire=120ms db_stagger=0ms total=47321ms prefetch_dedup=5/30",
            joined,
        )
        self.assertIn("[cycle] slowest id=1 eval=5000ms id=2 eval=3000ms id=3 eval=1000ms", joined)


class TestPrefetchDedup(unittest.TestCase):
    def setUp(self):
        _PRICE_TA_CACHE.clear()

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
    def setUp(self):
        _PRICE_TA_CACHE.clear()

    def test_cache_key_includes_asset_class_and_timeframe(self):
        key = _price_ta_cache_key("XAUUSD", "forex", "15m")
        self.assertEqual(key, "forex:XAUUSD:15m")

    def test_lookup_miss_when_empty(self):
        self.assertIsNone(
            _price_ta_cache_lookup("EURUSD", "forex", "15m"),
        )

    def test_metals_prefetch_hit_from_forex_eval_key(self):
        from datetime import datetime

        prefetched = {"price": 2650.0, "rsi": 50.0}
        _PRICE_TA_CACHE["metals:XAUUSD:15m"] = (
            prefetched,
            datetime.utcnow(),
        )
        strategy = type("S", (), {
            "asset_class": "forex",
            "config": {"asset_class": "forex", "universe": {"symbols": ["XAUUSD"]}},
        })()
        cache_ac = _strategy_cache_asset_class(strategy)
        hit = _price_ta_cache_lookup(
            "XAUUSD", cache_ac, "15m", metal_paper_ok=True,
        )
        self.assertIs(hit, prefetched)

    def test_metals_prefetch_hit_without_paper_suffix(self):
        from datetime import datetime

        prefetched = {"price": 2650.0}
        _PRICE_TA_CACHE["forex:XAUUSD:15m"] = (
            prefetched,
            datetime.utcnow(),
        )
        hit = _price_ta_cache_lookup(
            "XAUUSD", "metals", "15m", metal_paper_ok=True,
        )
        self.assertIs(hit, prefetched)


class TestEvalDiagLog(unittest.TestCase):
    def test_eval_diag_log_format(self):
        diag = EvalDiag(
            sem_wait_ms=12,
            db_slot_wait_ms=3,
            pool_wait_ms=1,
            db_ms=45,
            price_fetch_ms=18500,
            ta_ms=120,
            conditions_ms=800,
            cache_hits=0,
            cache_misses=2,
        )
        with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
            _log_eval_diag(42, diag, 19500.0)
        joined = "\n".join(cm.output)
        self.assertIn(
            "[eval] id=42 db=45ms price_fetch=18500ms ta=120ms conditions=800ms "
            "sem_wait=12ms db_slot_wait=3ms pool_wait=1ms cache_hits=0/2 total=19500ms",
            joined,
        )


class TestExecutorCycleCtx(unittest.TestCase):
    def test_shared_kline_cache_field(self):
        ctx = ExecutorCycleCtx(shard_index=1)
        ctx.kline_cache[("XAUUSD", "15m", 200, "forex")] = [1, 2, 3]
        self.assertEqual(len(ctx.kline_cache), 1)
