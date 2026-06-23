"""Stage B — tradfi split-eval: no DB session held across HTTP/conditions."""
from __future__ import annotations

import inspect
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.strategy_executor import (
    EvalDiag,
    _TradfiEvalCarry,
    _evaluate_tradfi_split,
    evaluate_and_fire,
    tradfi_eval_http_phase_active,
)


class TestForexDbSplitEval(unittest.TestCase):
    def test_forex_shard_uses_tradfi_split_helper(self):
        import app.services.strategy_executor as se

        src = inspect.getsource(se._run_forex_executor_shard)
        self.assertIn("_evaluate_tradfi_split(", src)
        self.assertNotIn(
            "evaluate_and_fire(\n                                            strategy, user, db_one",
            src,
        )

    def test_crypto_shard_preload_uses_threaded_db_phase(self):
        import app.services.strategy_executor as se

        src = inspect.getsource(se._run_crypto_executor_shard)
        self.assertIn("_run_db_phase_with_timeout(", src)
        self.assertIn('label=f"{_crypto_lbl}:preload_users"', src)
        self.assertIn('label=f"{_crypto_lbl}:gate_prefetch"', src)

    def test_evaluate_and_fire_accepts_split_params(self):
        sig = inspect.signature(evaluate_and_fire)
        self.assertIn("tradfi_db_split", sig.parameters)
        self.assertIn("gate_only", sig.parameters)
        self.assertIn("carry", sig.parameters)

    def test_gate_only_builds_carry(self):
        carry = _TradfiEvalCarry(
            strategy_row=object(),
            user_row=object(),
            strategy_id=1,
            strategy_name="t",
            strategy_status="active",
            config={"asset_class": "forex"},
            asset_class="forex",
            is_paper=False,
            early_fire_targets=[],
            candidate_symbols=["XAUUSD"],
            direction_pref="LONG",
            risk={},
            filters={},
            strictness_level=0,
            eval_tf="15m",
            uid=9,
            metal_paper=False,
            cache_ac="forex",
            strategy_gates={},
            gate_stats=None,
            diag=EvalDiag(),
            prefetched_ctrader_ok=True,
        )
        self.assertEqual(carry.strategy_id, 1)
        self.assertEqual(carry.candidate_symbols, ["XAUUSD"])

    def test_http_phase_contextvar(self):
        from app.services.strategy_executor import _TRADFI_EVAL_HTTP_PHASE

        tok = _TRADFI_EVAL_HTTP_PHASE.set(True)
        try:
            self.assertTrue(tradfi_eval_http_phase_active())
        finally:
            _TRADFI_EVAL_HTTP_PHASE.reset(tok)
        self.assertFalse(tradfi_eval_http_phase_active())


class TestForexDbSplitEvalAsync(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_tradfi_split_closes_gate_db_before_http(self):
        import app.services.strategy_executor as se

        snap = {"id": 42, "config": {"asset_class": "forex", "universe": {"symbols": ["XAUUSD"]}}}
        strategy_row = MagicMock()
        user_row = MagicMock()
        user_row.id = 7
        user_row.banned = False

        gate_carry = _TradfiEvalCarry(
            strategy_row=strategy_row,
            user_row=user_row,
            strategy_id=42,
            strategy_name="Gold",
            strategy_status="active",
            config={"asset_class": "forex", "universe": {"symbols": ["XAUUSD"]}},
            asset_class="forex",
            is_paper=False,
            early_fire_targets=[],
            candidate_symbols=["XAUUSD"],
            direction_pref="LONG",
            risk={},
            filters={},
            strictness_level=0,
            eval_tf="15m",
            uid=7,
            metal_paper=False,
            cache_ac="forex",
            strategy_gates={},
            gate_stats={},
            diag=EvalDiag(),
            prefetched_ctrader_ok=True,
        )

        closed = []

        class _FakeSession:
            def close(self):
                closed.append(True)

            def merge(self, row):
                return MagicMock(id=42, name="Gold", status="active", config=gate_carry.config)

        fake_db = _FakeSession()

        async def _fake_gate(*args, **kwargs):
            if kwargs.get("gate_only"):
                return gate_carry
            return None

        with patch("app.database.ForexSessionLocal", lambda: fake_db), patch(
            "app.database.forex_db_slot", return_value=_AsyncCM(),
        ), patch.object(se, "evaluate_and_fire", side_effect=_fake_gate), patch.object(
            se, "_evaluate_with_budget", new_callable=AsyncMock,
        ) as mock_budget:
            await _evaluate_tradfi_split(
                snap,
                strategy_row,
                user_row,
                MagicMock(),
                gate_stats={},
                prefetched_ctrader_ok=True,
                eval_diag=EvalDiag(),
            )
        self.assertTrue(closed, "gate session should close before HTTP resume")
        mock_budget.assert_awaited_once()
        resume_coro = mock_budget.await_args.args[3]
        self.assertTrue(hasattr(resume_coro, "cr_code"))


class _AsyncCM:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


if __name__ == "__main__":
    unittest.main()
