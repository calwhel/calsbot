"""Per-strategy eval sub-phase profiling and kline cache aliasing."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import logging
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.strategy_executor import (
    ExecutorCycleCtx,
    _EXECUTOR_CYCLE_CTX,
    _log_eval_phase_timing,
    _seed_kline_cache,
    _strategy_eval_fingerprint,
)


class TestEvalPhaseLogging(unittest.TestCase):
    def test_eval_log_format(self):
        with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
            _log_eval_phase_timing(
                214,
                db_caps_ms=12.0,
                price_fetch_ms=800.0,
                ta_compute_ms=5200.0,
                conditions_ms=5300.0,
                profile="syms=XAUUSD tfs=15m,1h htf=True nconds=5",
            )
        joined = "\n".join(cm.output)
        self.assertIn(
            "[eval] id=214 price_fetch=800ms ta_compute=5200ms "
            "db_caps=12ms conditions=5300ms",
            joined,
        )
        self.assertIn("syms=XAUUSD", joined)

    def test_eval_log_skips_fast(self):
        with patch("app.services.strategy_executor.logger") as mock_log:
            _log_eval_phase_timing(
                1, db_caps_ms=1, price_fetch_ms=2,
                ta_compute_ms=3, conditions_ms=4,
            )
        mock_log.info.assert_not_called()


class TestStrategyFingerprint(unittest.TestCase):
    def test_fingerprint_captures_multi_tf_and_htf(self):
        cfg = {
            "universe": {"symbols": ["XAUUSD", "EURUSD"]},
            "filters": {"htf_trend": True},
            "entry_conditions": {
                "conditions": [
                    {"type": "rsi", "timeframe": "15m"},
                    {"type": "fvg", "timeframe": "1h"},
                    {"type": "fx_killzone"},
                ],
            },
        }
        fp = _strategy_eval_fingerprint(cfg)
        self.assertIn("XAUUSD", fp)
        self.assertIn("15m", fp)
        self.assertIn("1h", fp)
        self.assertIn("htf=True", fp)
        self.assertIn("nconds=3", fp)


class TestKlineCacheSeeding(unittest.TestCase):
    def test_seed_populates_limit_aliases(self):
        cache: dict = {}
        klines = [[i, 1, 2, 3, 4, 5] for i in range(200)]
        _seed_kline_cache(cache, "XAUUSD", "forex", "15m", klines)
        self.assertIn(("XAUUSD", "15m", 50, "forex"), cache)
        self.assertIn(("XAUUSD", "15m", 200, "forex"), cache)
        self.assertEqual(len(cache[("XAUUSD", "15m", 50, "forex")]), 50)

    def test_get_klines_hits_larger_cached_limit(self):
        from app.services.strategy_ta import _get_klines

        ctx = ExecutorCycleCtx()
        ctx.kline_cache[("EURUSD", "15m", 200, "forex")] = [
            [i, 1, 2, 3, 4, 5] for i in range(200)
        ]
        token = _EXECUTOR_CYCLE_CTX.set(ctx)
        try:
            cache = {"__asset_class__": "forex"}
            cache.update(ctx.kline_cache)
            out = asyncio.run(
                _get_klines("EURUSD", "15m", 50, MagicMock(), cache),
            )
            self.assertEqual(len(out or []), 50)
            self.assertEqual(ctx.kline_cache_hits, 1)
            self.assertEqual(ctx.kline_cache_misses, 0)
        finally:
            _EXECUTOR_CYCLE_CTX.reset(token)


class TestCycleKlineCacheLog(unittest.TestCase):
    def test_cycle_log_includes_kline_hits(self):
        from app.services.strategy_executor import _log_cycle_timing

        ctx = ExecutorCycleCtx(kline_cache_hits=12, kline_cache_misses=3)
        token = _EXECUTOR_CYCLE_CTX.set(ctx)
        try:
            with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
                _log_cycle_timing(0, 10, 100, 5000, 0, 5100)
            joined = "\n".join(cm.output)
            self.assertIn("[cycle] kline_cache hits=12 misses=3", joined)
        finally:
            _EXECUTOR_CYCLE_CTX.reset(token)
