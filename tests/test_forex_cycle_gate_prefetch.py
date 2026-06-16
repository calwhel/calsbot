"""Cycle-level batch prefetch for cap/cooldown/assignment gates."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from app.services.strategy_executor import (
    ExecutorCycleCtx,
    _EXECUTOR_CYCLE_CTX,
    _daily_execution_count,
    _open_execution_count,
    _prefetch_cycle_gate_data,
    _prefetch_symbol_cooldowns,
)


class TestCycleGatePrefetch(unittest.TestCase):
    def test_daily_open_counts_use_cycle_ctx(self):
        ctx = ExecutorCycleCtx(execution_counts={42: (3, 1)})
        token = _EXECUTOR_CYCLE_CTX.set(ctx)
        try:
            db = MagicMock()
            self.assertEqual(_daily_execution_count(42, db), 3)
            self.assertEqual(_open_execution_count(42, db), 1)
            db.query.assert_not_called()
        finally:
            _EXECUTOR_CYCLE_CTX.reset(token)

    def test_symbol_cooldowns_use_cycle_ctx(self):
        ts = datetime.utcnow() - timedelta(minutes=10)
        ctx = ExecutorCycleCtx(
            symbol_last_fired={7: {"EURUSD": ts, "GBPUSD": ts}},
        )
        token = _EXECUTOR_CYCLE_CTX.set(ctx)
        try:
            fired_today, last_map = _prefetch_symbol_cooldowns(
                7, ["EURUSD", "XAUUSD"], MagicMock(), need_today=True,
            )
            self.assertIn("EURUSD", last_map)
            self.assertNotIn("XAUUSD", last_map)
            self.assertIn("EURUSD", fired_today)
        finally:
            _EXECUTOR_CYCLE_CTX.reset(token)

    def test_prefetch_cycle_gate_data_aggregates(self):
        """One prefetch call issues grouped queries, not per-strategy counts."""
        session_factory = MagicMock()
        db = MagicMock()
        session_factory.return_value = db

        count_row = MagicMock()
        count_row.strategy_id = 1
        count_row.daily_cnt = 2
        count_row.open_cnt = 1

        cooldown_row = MagicMock()
        cooldown_row.strategy_id = 1
        cooldown_row.symbol = "EURUSD"
        cooldown_row.last_at = datetime.utcnow()

        assign_row = MagicMock()
        assign_row.strategy_id = 1
        assign_row.ctrader_account_id = "12345"
        assign_row.enabled = True
        assign_row.lot_size = 0.1

        query_results = iter([
            [count_row],
            [cooldown_row],
            [assign_row],
        ])

        def _query(*_args, **_kwargs):
            q = MagicMock()
            q.filter.return_value = q
            q.group_by.return_value = q
            q.order_by.return_value = q
            q.all.return_value = next(query_results)
            return q

        db.query.side_effect = _query
        db.get_bind.return_value = MagicMock()

        with patch(
            "app.services.strategy_account_assignments.ensure_strategy_account_assignments_table",
        ):
            result = _prefetch_cycle_gate_data([1, 2], session_factory)

        self.assertEqual(result["execution_counts"][1], (2, 1))
        self.assertEqual(result["execution_counts"][2], (0, 0))
        self.assertIn("EURUSD", result["symbol_last_fired"][1])
        self.assertEqual(
            result["assignment_targets"][1][0]["ctrader_account_id"],
            "12345",
        )
        self.assertEqual(result["assignment_targets"][2], [])
        db.close.assert_called_once()


class TestAssignmentPrefetchIntegration(unittest.TestCase):
    def test_get_enabled_fire_targets_uses_prefetched_rows(self):
        from app.services.strategy_account_assignments import get_enabled_fire_targets

        strategy = MagicMock()
        strategy.id = 99
        strategy.ctrader_account_id = None
        prefetched = [{"ctrader_account_id": "999", "lot_size": 0.2}]
        out = get_enabled_fire_targets(
            MagicMock(), strategy, None, for_live_fire=True,
            prefetched_rows=prefetched,
        )
        self.assertEqual(out, prefetched)

    def test_resolve_live_fire_intent_with_prefetch(self):
        from app.services.strategy_account_assignments import resolve_live_fire_intent

        strategy = MagicMock()
        strategy.id = 5
        strategy.status = "active"
        strategy.ctrader_account_id = None
        wants_live, targets = resolve_live_fire_intent(
            MagicMock(), strategy, "forex", None,
            prefetched_targets=[{"ctrader_account_id": "111", "lot_size": None}],
        )
        self.assertTrue(wants_live)
        self.assertEqual(len(targets), 1)
