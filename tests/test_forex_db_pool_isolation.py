"""Stage A — dedicated forex_engine pool ring-fenced from bg_engine."""
from __future__ import annotations

import inspect
import os
import unittest

from app.database import (
    BgSessionLocal,
    ForexSessionLocal,
    bg_engine,
    bg_pool_hard_limit,
    forex_db_slot_limit,
    forex_engine,
    forex_engine_runtime_profile,
    forex_pool_hard_limit,
)


class TestForexDbPoolIsolation(unittest.TestCase):
    def test_forex_and_bg_engines_are_distinct(self):
        self.assertIsNot(forex_engine, bg_engine)
        self.assertEqual(str(forex_engine.url), str(bg_engine.url))

    def test_session_factories_bind_to_expected_engines(self):
        self.assertIs(BgSessionLocal.kw["bind"], bg_engine)
        self.assertIs(ForexSessionLocal.kw["bind"], forex_engine)

    def test_pool_hard_limits_are_independent(self):
        fx_limit = forex_pool_hard_limit()
        bg_limit = bg_pool_hard_limit()
        self.assertEqual(fx_limit, int(os.getenv("FOREX_POOL_SIZE", "4")) + int(os.getenv("FOREX_POOL_OVERFLOW", "4")))
        self.assertEqual(bg_limit, int(os.getenv("BG_POOL_SIZE", "8")) + int(os.getenv("BG_POOL_OVERFLOW", "10")))
        self.assertGreater(bg_limit, fx_limit)

    def test_forex_slot_limit_respects_pool_reserve(self):
        prof = forex_engine_runtime_profile()
        self.assertIn("pool_hard_limit", prof)
        self.assertIn("forex_db_slot_limit", prof)
        self.assertLessEqual(prof["forex_db_slot_limit"], prof["pool_hard_limit"])

    def test_forex_executor_shard_uses_forex_pool(self):
        import app.services.strategy_executor as se

        src = inspect.getsource(se._run_forex_executor_shard)
        self.assertIn("ForexSessionLocal as SessionLocal", src)
        self.assertIn("forex_engine as engine", src)
        self.assertIn("forex_engine_runtime_profile", src)
        self.assertIn("forex_db_slot", src)
        self.assertNotIn("BgSessionLocal as SessionLocal", src)
        self.assertNotIn("bg_engine as engine", src)

    def test_crypto_executor_shard_still_uses_bg_pool(self):
        import app.services.strategy_executor as se

        src = inspect.getsource(se._run_crypto_executor_shard)
        self.assertIn("BgSessionLocal as SessionLocal", src)
        self.assertNotIn("ForexSessionLocal", src)

    def test_forex_live_manager_paths_use_forex_pool(self):
        import app.services.strategy_executor as se

        for fn_name in (
            "_build_forex_worklist_impl",
            "_amend_forex_position_tick",
            "_close_live_forex_execution_and_notify",
            "_build_forex_reconcile_worklist",
        ):
            src = inspect.getsource(getattr(se, fn_name))
            self.assertIn("ForexSessionLocal", src, msg=fn_name)
            self.assertNotIn("BgSessionLocal", src, msg=fn_name)

        worklist_src = inspect.getsource(se._build_forex_worklist)
        self.assertIn("forex_engine", worklist_src)

    def test_forex_db_slot_limit_default_within_pool(self):
        limit = forex_db_slot_limit()
        hard = forex_pool_hard_limit()
        self.assertGreaterEqual(limit, 2)
        self.assertLessEqual(limit, hard)


if __name__ == "__main__":
    unittest.main()
