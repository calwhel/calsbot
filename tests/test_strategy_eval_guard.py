"""Per-strategy eval guards — universe cap, symbol normalization, eval budget."""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio

from app.services.strategy_executor import (
    EXECUTOR_MAX_SYMBOLS_PER_STRATEGY,
    EXECUTOR_STRATEGY_EVAL_BUDGET_S,
    _PRICE_TA_CACHE,
    _evaluate_with_budget,
    _normalize_universe_symbol,
    _price_ta_cache_lookup,
    _symbols_for_snapshot,
    _tradfi_universe_raw_count,
    _tradfi_universe_symbols,
)


class TestUniverseNormalization(unittest.TestCase):
    def test_normalize_strips_slashes_and_dashes(self):
        self.assertEqual(_normalize_universe_symbol("eur/usd"), "EURUSD")
        self.assertEqual(_normalize_universe_symbol("XAU-USD"), "XAUUSD")

    def test_tradfi_universe_caps_symbol_count(self):
        cfg = {
            "universe": {
                "type": "specific",
                "symbols": [f"PAIR{i}" for i in range(30)],
            },
        }
        syms = _tradfi_universe_symbols(cfg, max_symbols=20)
        self.assertEqual(len(syms), 20)
        self.assertEqual(_tradfi_universe_raw_count(cfg), 30)

    def test_prefetch_and_eval_share_symbol_helper(self):
        snap = {
            "config": {
                "universe": {
                    "type": "specific",
                    "symbols": ["EUR/USD", "GBP-USD", "XAUUSD"],
                },
            },
        }
        self.assertEqual(
            _symbols_for_snapshot(snap),
            ["EURUSD", "GBPUSD", "XAUUSD"],
        )


class TestCacheKeyNormalization(unittest.TestCase):
    def setUp(self):
        _PRICE_TA_CACHE.clear()

    def test_eval_hits_prefetch_key_after_normalization(self):
        from datetime import datetime

        prefetched = {"price": 1.085}
        _PRICE_TA_CACHE["forex:EURUSD:15m"] = (prefetched, datetime.utcnow())
        hit = _price_ta_cache_lookup("EUR/USD", "forex", "15m")
        self.assertIs(hit, prefetched)


class TestEvalBudget(unittest.IsolatedAsyncioTestCase):
    async def test_abort_logs_when_budget_exceeded(self):
        async def _slow():
            await asyncio.sleep(0.2)

        import app.services.strategy_executor as se
        with patch.object(se, "EXECUTOR_STRATEGY_EVAL_BUDGET_S", 0.05), \
                self.assertLogs("app.services.strategy_executor", level="WARNING") as cm:
            cfg = {"universe": {"symbols": ["EURUSD"] * 30}}
            with self.assertRaises(asyncio.TimeoutError):
                await _evaluate_with_budget(86, cfg, _slow())
        self.assertTrue(
            any("[eval] id=86 ABORTED >budget" in line for line in cm.output),
        )

    def test_default_budget_is_ten_seconds(self):
        self.assertEqual(EXECUTOR_STRATEGY_EVAL_BUDGET_S, 10.0)
        self.assertEqual(EXECUTOR_MAX_SYMBOLS_PER_STRATEGY, 20)


if __name__ == "__main__":
    unittest.main()
