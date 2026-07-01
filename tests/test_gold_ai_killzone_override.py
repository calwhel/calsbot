"""Killzone quality override — env helpers, funnel, and loop source contracts."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gold_ai_trader.call_gates import (
    candidate_confluence_counts,
    candidate_meets_killzone_override,
    killzone_override_enabled,
    killzone_override_min_confluence,
)
from app.gold_ai_trader.funnel import record, snapshot


ROOT = Path(__file__).resolve().parents[1]
LOOP = (ROOT / "app/gold_ai_trader/loop.py").read_text(encoding="utf-8")
CALL_GATES = (ROOT / "app/gold_ai_trader/call_gates.py").read_text(encoding="utf-8")


def _candidate(checklist: dict):
    return SimpleNamespace(raw={"readiness_checklist": checklist})


class KillzoneOverrideEnvTests(unittest.TestCase):
    def setUp(self):
        self._saved = {
            "GOLD_AI_KILLZONE_OVERRIDE_ENABLED": os.environ.get(
                "GOLD_AI_KILLZONE_OVERRIDE_ENABLED"
            ),
            "GOLD_AI_KILLZONE_OVERRIDE_MIN_CONFLUENCE": os.environ.get(
                "GOLD_AI_KILLZONE_OVERRIDE_MIN_CONFLUENCE"
            ),
        }

    def tearDown(self):
        for key, val in self._saved.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val

    def test_override_disabled_by_default(self):
        os.environ.pop("GOLD_AI_KILLZONE_OVERRIDE_ENABLED", None)
        self.assertFalse(killzone_override_enabled())

    def test_override_enabled_when_set(self):
        os.environ["GOLD_AI_KILLZONE_OVERRIDE_ENABLED"] = "true"
        self.assertTrue(killzone_override_enabled())

    def test_min_confluence_default_six(self):
        os.environ.pop("GOLD_AI_KILLZONE_OVERRIDE_MIN_CONFLUENCE", None)
        self.assertEqual(killzone_override_min_confluence(), 6)

    def test_candidate_meets_threshold_passed_gte_min(self):
        os.environ["GOLD_AI_KILLZONE_OVERRIDE_MIN_CONFLUENCE"] = "5"
        checklist = {
            "htf_aligned": True,
            "at_entry": True,
            "displacement_ok": True,
            "momentum_ok": True,
            "rr_feasible": True,
            "premium_discount_ok": False,
        }
        passed, total = candidate_confluence_counts(_candidate(checklist))
        self.assertEqual(passed, 5)
        self.assertEqual(total, 6)
        self.assertTrue(candidate_meets_killzone_override(_candidate(checklist)))

    def test_candidate_below_threshold_fails(self):
        os.environ["GOLD_AI_KILLZONE_OVERRIDE_MIN_CONFLUENCE"] = "5"
        checklist = {
            "htf_aligned": True,
            "at_entry": True,
            "displacement_ok": True,
            "momentum_ok": False,
            "rr_feasible": True,
            "premium_discount_ok": False,
        }
        self.assertFalse(candidate_meets_killzone_override(_candidate(checklist)))


class KillzoneOverrideFunnelTests(unittest.TestCase):
    def test_override_confluence_skipped_event(self):
        record("override_confluence_skipped", reason="confluence_3/6<5")
        snap = snapshot()
        self.assertGreaterEqual(snap.get("override_confluence_skipped", 0), 1)


class KillzoneOverrideLoopSourceTests(unittest.TestCase):
    def test_gate_a_allows_override_without_orb(self):
        self.assertIn(
            "if killzone_blocked and not orb_enabled and not killzone_override_enabled():",
            LOOP,
        )

    def test_gate_b_blocks_only_when_override_off(self):
        self.assertIn(
            "if killzone_blocked and not killzone_override_enabled():",
            LOOP,
        )

    def test_override_scans_before_claude_path(self):
        override_idx = LOOP.index("killzone_override_scan = killzone_blocked")
        scan_idx = LOOP.index("price, candidates = await scan_candidates(", override_idx)
        self.assertLess(override_idx, scan_idx)

    def test_override_confluence_gate_before_dedupe(self):
        conf_idx = LOOP.index('funnel_record(\n                        "override_confluence_skipped",')
        dedupe_idx = LOOP.index("ok_dedupe, dedupe_reason = should_invoke_claude(", conf_idx)
        self.assertLess(conf_idx, dedupe_idx)

    def test_distinct_dormant_reasons(self):
        self.assertIn('note_dormant("outside_killzone")', LOOP)
        self.assertIn('note_dormant("outside_killzone_low_confluence")', LOOP)

    def test_claude_budget_checked_for_override_scan(self):
        self.assertIn("if not killzone_blocked or killzone_override_scan:", LOOP)
        self.assertIn("ok_call, reason = check_can_call_claude(db, cfg)", LOOP)

    def test_in_killzone_path_unchanged_when_override_off(self):
        """Toggle off: Gate B still uses outside_killzone; no override scan variable in gate A."""
        self.assertIn("killzone_override_enabled()", CALL_GATES)
        self.assertIn('os.environ.get("GOLD_AI_KILLZONE_OVERRIDE_ENABLED", "false")', CALL_GATES)

    def test_data_and_news_gates_before_override_scan(self):
        data_idx = LOOP.index("data_ok, data_block = gold_data_ok_for_claude(market_data)")
        news_idx = LOOP.index('funnel_record(\n                    "news_blocked"', data_idx)
        scan_idx = LOOP.index("price, candidates = await scan_candidates(", news_idx)
        self.assertLess(data_idx, scan_idx)
        self.assertLess(news_idx, scan_idx)


if __name__ == "__main__":
    unittest.main()
