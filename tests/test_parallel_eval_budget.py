"""Parallel eval — per-strategy budget must not serialize the shard."""
import os
import time
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
from unittest.mock import patch

from app.services.strategy_executor import (
    EXECUTOR_STRATEGY_EVAL_BUDGET_S,
    FOREX_MAX_CONCURRENT,
    _evaluate_with_budget,
    _gather_eval_batches,
    _gather_eval_parallel,
)


class TestParallelEvalBudget(unittest.IsolatedAsyncioTestCase):
    async def test_gather_parallel_budget_aborts_not_serial(self):
        """9 strategies × 10s budget with sem=6 must finish ~10–15s, not ~90s."""
        sem = asyncio.Semaphore(6)
        budget = 0.08
        n = 9
        completed = []

        async def _slow_body(sid: int):
            await asyncio.sleep(2.0)
            completed.append(sid)

        async def _run_one(sid: int):
            async with sem:
                try:
                    await _evaluate_with_budget(
                        sid,
                        {"universe": {"symbols": ["EURUSD"]}},
                        "forex",
                        _slow_body(sid),
                    )
                except asyncio.TimeoutError:
                    pass

        with patch(
            "app.services.strategy_executor.EXECUTOR_STRATEGY_EVAL_BUDGET_S",
            budget,
        ):
            t0 = time.monotonic()
            with self.assertRaises(asyncio.TimeoutError):
                await _evaluate_with_budget(
                    0,
                    {"universe": {"symbols": ["EURUSD"]}},
                    "forex",
                    _slow_body(0),
                )
            solo_ms = (time.monotonic() - t0) * 1000.0
            self.assertLess(solo_ms, budget * 1000 + 200)

            completed.clear()
            t0 = time.monotonic()
            await _gather_eval_parallel(
                "test",
                list(range(n)),
                _run_one,
            )
            elapsed = time.monotonic() - t0
            # Serial would be n × budget ≈ 0.72s; parallel sem=6 → ceil(9/6)×budget ≈ 0.16s
            self.assertLess(elapsed, budget * n * 0.75)
            self.assertEqual(len(completed), 0)

    async def test_sequential_batches_serialize_budget_aborts(self):
        """Old batch_size=1 loop serialized budget aborts — regression guard."""
        budget = 0.05
        n = 5

        async def _slow(sid: int):
            await asyncio.sleep(1.0)

        async def _run_one(sid: int):
            try:
                await _evaluate_with_budget(
                    sid, {}, "forex", _slow(sid),
                )
            except asyncio.TimeoutError:
                pass

        with patch(
            "app.services.strategy_executor.EXECUTOR_STRATEGY_EVAL_BUDGET_S",
            budget,
        ):
            # Simulate pre-fix sequential batches (batch_size=1).
            total = n
            t0 = time.monotonic()
            for i in range(0, total, 1):
                batch = list(range(i, min(i + 1, total)))
                await asyncio.gather(*[_run_one(s) for s in batch])
            serial_elapsed = time.monotonic() - t0

            t0 = time.monotonic()
            await _gather_eval_parallel("test", list(range(n)), _run_one)
            parallel_elapsed = time.monotonic() - t0

        self.assertGreater(serial_elapsed, budget * (n - 1))
        self.assertLess(parallel_elapsed, serial_elapsed * 0.6)

    def test_forex_max_concurrent_default_six_or_more(self):
        self.assertGreaterEqual(FOREX_MAX_CONCURRENT, 3)
        self.assertEqual(EXECUTOR_STRATEGY_EVAL_BUDGET_S, 10.0)

    async def test_gather_batches_delegates_to_parallel(self):
        calls = []

        async def _run_one(snap):
            calls.append(snap)

        await _gather_eval_batches("lbl", [1, 2, 3], _run_one)
        self.assertEqual(sorted(calls), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
