"""Crypto disable flag + per-cycle gold condition kline cache."""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio

from app.services.strategy_executor import (
    _prefetch_cycle_condition_klines,
    crypto_executor_disabled,
    run_strategy_executor,
)
from app.services.strategy_ta import collect_condition_kline_timeframes


class TestCryptoExecutorDisabled(unittest.IsolatedAsyncioTestCase):
    async def test_run_strategy_executor_returns_immediately_when_disabled(self):
        with patch.dict(os.environ, {"DISABLE_CRYPTO_EXECUTOR": "1"}, clear=False):
            with patch(
                "app.strategy_models.init_strategy_tables",
            ) as mock_init:
                await run_strategy_executor()
                mock_init.assert_not_called()

    async def test_crypto_shard_not_started_when_disabled(self):
        from app.services.strategy_executor import _run_crypto_executor_shard

        http = MagicMock()
        with patch.dict(os.environ, {"DISABLE_CRYPTO_EXECUTOR": "1"}, clear=False):
            await _run_crypto_executor_shard(0, 1, http)

    async def test_prefetch_price_ta_not_called_from_disabled_crypto_path(self):
        """Crypto prefetch only lives inside _run_crypto_executor_shard."""
        with patch.dict(os.environ, {"DISABLE_CRYPTO_EXECUTOR": "1"}, clear=False):
            self.assertTrue(crypto_executor_disabled())
            with patch(
                "app.services.strategy_executor._prefetch_price_ta_for_cycle",
                new_callable=AsyncMock,
            ) as mock_prefetch:
                await run_strategy_executor()
                mock_prefetch.assert_not_called()


class TestConditionKlineTimeframes(unittest.TestCase):
    def test_collects_smc_and_fx_timeframes(self):
        conds = [
            {"type": "order_block", "timeframe": "15m"},
            {"type": "fx_cisd", "timeframe": "5m"},
            {"type": "fx_killzone"},
            {"type": "fvg", "timeframe": "15m"},
        ]
        tfs = collect_condition_kline_timeframes(conds)
        self.assertEqual(tfs, ["15m", "5m"])

    def test_smc_subset_matches_legacy(self):
        from app.services.strategy_ta import _smc_prefetch_timeframes

        conds = [
            {"type": "order_block", "timeframe": "15m"},
            {"type": "fx_cisd", "timeframe": "5m"},
        ]
        self.assertEqual(_smc_prefetch_timeframes(conds), ["15m"])


class TestCycleConditionKlineCache(unittest.IsolatedAsyncioTestCase):
    async def test_warms_once_per_symbol_across_strategies(self):
        snapshots = [
            {
                "id": 200,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["XAUUSD"]},
                    "entry_conditions": {
                        "conditions": [
                            {"type": "order_block", "timeframe": "15m"},
                            {"type": "fx_cisd", "timeframe": "5m"},
                        ],
                    },
                },
            },
            {
                "id": 225,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["XAUUSD"]},
                    "entry_conditions": {
                        "conditions": [
                            {"type": "fvg", "timeframe": "15m"},
                            {"type": "market_structure", "timeframe": "15m"},
                        ],
                    },
                },
            },
        ]
        kline_cache: dict = {}
        fetch_calls: list = []

        async def _fake_get_klines(symbol, interval, limit, http_client, cache=None):
            fetch_calls.append((symbol, interval, limit))
            rows = [[i, 1, 1, 1, 1, 1] for i in range(200)]
            if cache is not None:
                ac = cache.get("__asset_class__", "forex")
                cache[(symbol, interval, limit, ac)] = rows
            return rows

        with patch(
            "app.services.strategy_ta._get_klines",
            side_effect=_fake_get_klines,
        ):
            stats = await _prefetch_cycle_condition_klines(
                snapshots,
                MagicMock(),
                kline_cache,
            )

        self.assertEqual(stats["symbol_keys"], 1)
        self.assertEqual(stats["cold_fetches"], 2)
        self.assertEqual(len(fetch_calls), 2)
        intervals = sorted(c[1] for c in fetch_calls)
        self.assertEqual(intervals, ["15m", "5m"])
        self.assertIn(("XAUUSD", "15m", 200, "forex"), kline_cache)
        self.assertIn(("XAUUSD", "5m", 200, "forex"), kline_cache)

        fetch_calls.clear()
        stats2 = await _prefetch_cycle_condition_klines(
            snapshots,
            MagicMock(),
            kline_cache,
        )
        self.assertEqual(stats2["cold_fetches"], 0)
        self.assertEqual(fetch_calls, [])


if __name__ == "__main__":
    unittest.main()
