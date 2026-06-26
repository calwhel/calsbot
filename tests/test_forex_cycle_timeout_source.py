import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = (ROOT / "app/services/strategy_executor.py").read_text(encoding="utf-8")
CTRADER_FEED = (ROOT / "app/services/ctrader_price_feed.py").read_text(encoding="utf-8")


class ForexCycleTimeoutSourceTests(unittest.TestCase):
    def test_forex_cycle_and_phase_timeout_guards_exist(self):
        self.assertIn("EXECUTOR_FOREX_CYCLE_TIMEOUT_S", EXECUTOR)
        self.assertIn("EXECUTOR_FOREX_PREFETCH_TIMEOUT_S", EXECUTOR)
        self.assertIn("EXECUTOR_FOREX_EVAL_PHASE_TIMEOUT_S", EXECUTOR)
        self.assertIn("def _phase_timeout_budget_s(", EXECUTOR)
        self.assertIn("timeout=_phase_budget(EXECUTOR_FOREX_PREFETCH_TIMEOUT_S)", EXECUTOR)
        self.assertIn("timeout=_phase_budget(EXECUTOR_FOREX_EVAL_PHASE_TIMEOUT_S)", EXECUTOR)

    def test_cycle_timeout_blocks_fire_path(self):
        self.assertIn("cycle_deadline_mono", EXECUTOR)
        self.assertIn("fire_blocked", EXECUTOR)
        self.assertIn("blk_cycle_timeout", EXECUTOR)
        self.assertIn("fire blocked — cycle timeout guard", EXECUTOR)

    def test_gate_only_phase_timeout_exists(self):
        self.assertIn("EXECUTOR_GATE_ONLY_TIMEOUT_S", EXECUTOR)
        self.assertIn("gate_only timeout", EXECUTOR)
        self.assertIn("gate_only=True", EXECUTOR)
        self.assertIn("await asyncio.wait_for(", EXECUTOR)

    def test_ctrader_account_lookup_is_threaded_bounded_and_cached(self):
        self.assertIn("CTRADER_ACCOUNT_LOOKUP_TIMEOUT_S", CTRADER_FEED)
        self.assertIn("CTRADER_ACCOUNT_LOOKUP_CACHE_TTL_S", CTRADER_FEED)
        self.assertIn("_connected_accounts_cache", CTRADER_FEED)
        self.assertIn("asyncio.to_thread(_list_connected_accounts_sync, user_id)", CTRADER_FEED)
        self.assertIn("DB lookup timeout", CTRADER_FEED)


if __name__ == "__main__":
    unittest.main()
