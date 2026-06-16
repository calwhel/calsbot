"""Forex executor loop resilience — heartbeats, restart wrapper, assignment guards."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import inspect
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.strategy_executor import (
    _aggregate_loop_name,
    get_heartbeats,
    mark_heartbeat,
    run_forex_executor,
)


class TestAggregateHeartbeats(unittest.TestCase):
    def test_shard_updates_aggregate_forex_executor(self):
        from app.services import strategy_executor as se

        se._EXECUTOR_HEARTBEATS.clear()
        mark_heartbeat("forex_executor_s1")
        hb = get_heartbeats()
        self.assertIn("forex_executor_s1", hb)
        self.assertIn("forex_executor", hb)
        self.assertEqual(hb["forex_executor"], hb["forex_executor_s1"])

    def test_aggregate_loop_name_mapping(self):
        self.assertEqual(_aggregate_loop_name("forex_executor_s0"), "forex_executor")
        self.assertEqual(_aggregate_loop_name("crypto_executor_s2"), "crypto_executor")
        self.assertIsNone(_aggregate_loop_name("paper_monitor"))


class TestForexExecutorRestartWrapper(unittest.TestCase):
    def test_run_forex_executor_has_critical_restart_loop(self):
        src = inspect.getsource(run_forex_executor)
        self.assertIn("forex_executor loop started", src)
        self.assertIn("logger.critical", src)
        self.assertIn("_FOREX_EXECUTOR_RESTART_SECS", src)
        self.assertIn("while True", src)

    def test_shard_has_heartbeat_loop(self):
        from app.services import strategy_executor as se

        src = inspect.getsource(se._run_forex_executor_shard)
        self.assertIn("_forex_executor_heartbeat_loop", src)
        self.assertIn("forex_executor loop started shard=", src)

    def test_restarts_after_shard_crash(self):
        calls = {"n": 0}

        async def _boom_shard(*_a, **_k):
            calls["n"] += 1
            if calls["n"] < 2:
                raise RuntimeError("simulated startup failure")

        async def _run():
            with patch.dict(os.environ, {"DISABLE_FOREX_EXECUTOR": "0"}, clear=False):
                with patch(
                    "app.services.strategy_executor._run_forex_executor_shard",
                    side_effect=_boom_shard,
                ):
                    with patch(
                        "app.services.strategy_executor._FOREX_EXECUTOR_RESTART_SECS",
                        0.01,
                    ):
                        with patch(
                            "app.services.strategy_executor.EXECUTOR_SHARD_COUNT",
                            1,
                        ):
                            task = asyncio.create_task(run_forex_executor())
                            await asyncio.sleep(0.08)
                            task.cancel()
                            try:
                                await task
                            except asyncio.CancelledError:
                                pass

        asyncio.run(_run())
        self.assertGreaterEqual(calls["n"], 2)


class TestAssignmentLookupGuard(unittest.TestCase):
    def test_get_enabled_fire_targets_survives_missing_table(self):
        from app.services.strategy_account_assignments import get_enabled_fire_targets

        db = MagicMock()
        db.get_bind.return_value = MagicMock()
        strategy = MagicMock(id=42, ctrader_account_id="12345", ctrader_account_lot=0.1)
        prefs = None

        with patch(
            "app.services.strategy_account_assignments.ensure_strategy_account_assignments_table",
            side_effect=RuntimeError("DDL failed"),
        ):
            targets = get_enabled_fire_targets(db, strategy, prefs)
        self.assertEqual(targets, [{"ctrader_account_id": "12345", "lot_size": 0.1}])


if __name__ == "__main__":
    unittest.main()
