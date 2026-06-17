"""Executor advisory-lock resilience and fail-closed fire gate guards."""

from __future__ import annotations

from pathlib import Path
import unittest


class _FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def execute(self, _sql, _params=None):
        return None

    def fetchone(self):
        return (12345,)


class _FakeConn:
    def cursor(self):
        return _FakeCursor()


class TestExecutorLeadershipState(unittest.TestCase):
    def test_uncertain_state_fails_closed(self):
        from app.executor_leadership import (
            executor_can_run,
            mark_executor_lock_acquired,
            mark_executor_lock_lost,
            mark_executor_lock_uncertain,
        )

        mark_executor_lock_lost("test-reset")
        self.assertFalse(executor_can_run())

        ok = mark_executor_lock_acquired(
            _FakeConn(),
            lock_id=708110000,
            application_name="th-executor",
            reason="test-acquire",
        )
        self.assertTrue(ok)
        self.assertTrue(executor_can_run())

        mark_executor_lock_uncertain("test-drop")
        self.assertFalse(executor_can_run())

    def test_live_check_fails_closed_without_confirmed_owner(self):
        from app.executor_leadership import (
            mark_executor_lock_lost,
            verify_executor_lock_live,
        )

        mark_executor_lock_lost("test-reset")
        self.assertFalse(verify_executor_lock_live())


class TestExecutorLockFlowWiring(unittest.TestCase):
    def test_keepalive_marks_uncertain_and_lost(self):
        src = Path("strategy_portal_server.py").read_text(encoding="utf-8")
        self.assertIn("mark_executor_lock_uncertain(", src)
        self.assertIn("mark_executor_lock_lost(", src)
        self.assertIn("mark_executor_lock_acquired(", src)

    def test_executor_loops_gate_on_lock_ownership(self):
        src = Path("app/services/strategy_executor.py").read_text(encoding="utf-8")
        self.assertIn("if not _executor_owner_confirmed()", src)
        self.assertIn("async def _run_crypto_executor_shard", src)
        self.assertIn("async def _run_forex_executor_shard", src)

    def test_fire_paths_use_live_lock_gate(self):
        src_eval = Path("app/services/strategy_executor.py").read_text(encoding="utf-8")
        self.assertIn("_verify_executor_fire_gate(", src_eval)

        src_queue = Path("app/services/ctrader_order_queue.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("_lock_gate_allows_broker_send(job)", src_queue)

        trader_src = Path("app/services/strategy_trader.py").read_text(encoding="utf-8")
        self.assertIn("verify_executor_lock_live", trader_src)

    def test_paper_guard_not_regressed(self):
        src = Path("app/services/strategy_account_assignments.py").read_text(
            encoding="utf-8",
        )
        self.assertIn("status_active = status == \"active\"", src)
        self.assertIn("if not status_active:", src)
        self.assertIn("return False, []", src)


if __name__ == "__main__":
    unittest.main()
