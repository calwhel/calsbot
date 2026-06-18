"""Per-strategy eval guards — universe cap, symbol normalization, eval budget."""
import inspect
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio

from app.services.strategy_executor import (
    EXECUTOR_MAX_SYMBOLS_PER_STRATEGY,
    EXECUTOR_STRATEGY_EVAL_BUDGET_S,
    _PRICE_TA_CACHE,
    executor_runtime_profile,
    _gather_eval_batches,
    _evaluate_with_budget,
    _has_empty_specific_universe,
    _normalize_universe_symbol,
    _pre_eval_skip_no_symbols,
    _price_ta_cache_lookup,
    _symbols_for_snapshot,
    _tradfi_universe_raw_count,
    _tradfi_universe_symbols,
)
from app.services.strategy_ta import StrategyEvalCancelled


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


class TestEmptyUniverseSkip(unittest.TestCase):
    def test_empty_specific_forex_universe(self):
        cfg = {"universe": {"type": "specific", "symbols": []}, "asset_class": "forex"}
        self.assertTrue(_has_empty_specific_universe(cfg, "forex"))

    def test_crypto_all_is_not_empty_specific(self):
        cfg = {"universe": {"type": "all"}}
        self.assertFalse(_has_empty_specific_universe(cfg, "crypto"))

    def test_pre_eval_skip_logs(self):
        snap = {
            "id": 42,
            "config": {"universe": {"type": "specific", "symbols": []}, "asset_class": "forex"},
            "_obj": type("O", (), {"asset_class": "forex"})(),
        }
        with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
            self.assertTrue(_pre_eval_skip_no_symbols(snap))
        self.assertTrue(any("[eval] id=42 no symbols — skipped" in line for line in cm.output))


class TestEvalBudget(unittest.IsolatedAsyncioTestCase):
    async def test_abort_logs_when_budget_exceeded(self):
        async def _slow():
            await asyncio.sleep(0.2)

        import app.services.strategy_executor as se
        with patch.object(se, "EXECUTOR_STRATEGY_EVAL_BUDGET_S", 0.05), \
                self.assertLogs("app.services.strategy_executor", level="WARNING") as cm:
            cfg = {"universe": {"symbols": ["EURUSD"] * 30}}
            with self.assertRaises(asyncio.TimeoutError):
                await _evaluate_with_budget(86, cfg, "forex", _slow())
        self.assertTrue(
            any("[eval] id=86 ABORTED >budget" in line for line in cm.output),
        )

    def test_default_budget_is_ten_seconds(self):
        self.assertEqual(EXECUTOR_STRATEGY_EVAL_BUDGET_S, 10.0)
        self.assertEqual(EXECUTOR_MAX_SYMBOLS_PER_STRATEGY, 20)

    async def test_cancelled_eval_is_counted_as_budget_abort(self):
        async def _swallow_cancel():
            try:
                await asyncio.sleep(0.2)
            except asyncio.CancelledError as exc:
                raise StrategyEvalCancelled("EURUSD") from exc

        import app.services.strategy_executor as se
        with patch.object(se, "EXECUTOR_STRATEGY_EVAL_BUDGET_S", 0.05), \
                self.assertLogs("app.services.strategy_executor", level="WARNING") as cm:
            cfg = {"universe": {"symbols": ["EURUSD"]}}
            with self.assertRaises(asyncio.TimeoutError):
                await _evaluate_with_budget(87, cfg, "forex", _swallow_cancel())
        self.assertTrue(
            any(
                "[eval] id=87 ABORTED >budget" in line and "cause=cancelled_eval" in line
                for line in cm.output
            ),
        )


class TestEvalBatchLaunch(unittest.IsolatedAsyncioTestCase):
    async def test_gather_eval_batches_launches_all_tasks_together(self):
        snapshots = list(range(8))
        state = {"active": 0, "max_active": 0}
        lock = asyncio.Lock()

        async def _run_one(_snap):
            async with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            await asyncio.sleep(0.02)
            async with lock:
                state["active"] -= 1

        # Even with batch_size=1, launcher should still schedule all tasks at once.
        await _gather_eval_batches("test", snapshots, _run_one, batch_size=1)
        self.assertGreaterEqual(state["max_active"], 2)


class TestExecutorRuntimeProfile(unittest.TestCase):
    def test_profile_exposes_concurrency_and_provider_caps(self):
        profile = executor_runtime_profile()
        self.assertIn("forex_max_concurrent", profile)
        self.assertIn("executor_shard_count", profile)
        self.assertIn("strategy_eval_budget_s", profile)
        self.assertIn("prefetch_provider_limits", profile)
        self.assertIsInstance(profile["prefetch_provider_limits"], dict)
        self.assertIn("kraken", profile["prefetch_provider_limits"])
        self.assertIn("fmp", profile["prefetch_provider_limits"])


class TestExecutorDbResilienceGuards(unittest.TestCase):
    def test_forex_shard_init_is_timeout_guarded(self):
        import app.services.strategy_executor as se

        src = inspect.getsource(se._run_forex_executor_shard)
        self.assertIn("asyncio.wait_for(", src)
        self.assertIn("asyncio.to_thread(init_strategy_tables, engine)", src)
        self.assertIn("EXECUTOR_INIT_DB_TIMEOUT_S", src)

    def test_cycle_db_phases_use_threaded_timeout_wrapper(self):
        import app.services.strategy_executor as se

        src = inspect.getsource(se._run_forex_executor_shard)
        self.assertIn("_run_db_phase_with_timeout(", src)
        self.assertIn("label=f\"{_fx_lbl}:snapshots\"", src)
        self.assertIn("label=f\"{_fx_lbl}:preload_users\"", src)
        self.assertIn("label=f\"{_fx_lbl}:gate_prefetch\"", src)

    def test_bg_engine_uses_pre_ping_keepalive_profile(self):
        src = Path("app/database.py").read_text(encoding="utf-8")
        self.assertIn("pool_pre_ping=True", src)
        self.assertIn("application_name=APP_NAME_EXECUTOR", src)
        self.assertIn("BG_DB_KEEPALIVES_IDLE_S", src)
        self.assertIn("BG_DB_STATEMENT_TIMEOUT_MS", src)


if __name__ == "__main__":
    unittest.main()
