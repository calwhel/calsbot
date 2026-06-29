import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOOP = (ROOT / "app/gold_ai_trader/loop.py").read_text(encoding="utf-8")


class GoldAiOrbKillzoneDecoupleSourceTests(unittest.TestCase):
    def test_killzone_gate_kept_for_ict_but_not_orb(self):
        self.assertIn(
            "killzone_blocked = killzone_only_enabled() and not in_killzone(now, session, cfg)",
            LOOP,
        )
        self.assertIn("if killzone_blocked and not orb_enabled and not killzone_override_enabled():", LOOP)
        self.assertIn("if killzone_blocked and not killzone_override_enabled():", LOOP)
        self.assertIn("if not killzone_blocked or killzone_override_scan:", LOOP)
        self.assertIn("ok_call, reason = check_can_call_claude(db, cfg)", LOOP)
        self.assertIn("orb_logged = await _maybe_run_orb_strategy(", LOOP)
        self.assertIn("runtime_state.note_dormant(\"outside_killzone\")", LOOP)
        self.assertIn("outside_killzone_low_confluence", LOOP)

    def test_orb_runs_before_killzone_idle_return_and_before_ict_scan(self):
        orb_idx = LOOP.index("orb_logged = await _maybe_run_orb_strategy(")
        killzone_idle_idx = LOOP.index("if killzone_blocked and not killzone_override_enabled():", orb_idx)
        ict_scan_idx = LOOP.index("price, candidates = await scan_candidates(", orb_idx)
        self.assertLess(orb_idx, killzone_idle_idx)
        self.assertLess(killzone_idle_idx, ict_scan_idx)


if __name__ == "__main__":
    unittest.main()
