import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = (ROOT / "app/gold_ai_trader/scanner.py").read_text(encoding="utf-8")
TOGGLES = (ROOT / "app/gold_ai_trader/setup_toggles.py").read_text(encoding="utf-8")
PLAYBOOK = (ROOT / "app/gold_ai_trader/session_playbook.py").read_text(encoding="utf-8")
TRENDLINE = (ROOT / "app/gold_ai_trader/trendline.py").read_text(encoding="utf-8")
LOOP = (ROOT / "app/gold_ai_trader/loop.py").read_text(encoding="utf-8")
VALIDATOR = (ROOT / "app/gold_ai_trader/decision_validator.py").read_text(encoding="utf-8")


class GoldAiTrendlineSourceTests(unittest.TestCase):
    def test_scanner_wires_trendline_setups(self):
        self.assertIn("from app.gold_ai_trader.trendline import eval_fx_trendline", SCANNER)
        self.assertIn('"trendline_bounce_long"', SCANNER)
        self.assertIn('"trendline_bounce_short"', SCANNER)
        self.assertIn('"trendline_break_long"', SCANNER)
        self.assertIn('"trendline_break_short"', SCANNER)
        self.assertIn('"fx_trendline"', SCANNER)

    def test_global_toggle_default_off_and_env_keys_exist(self):
        self.assertIn("def trendline_enabled()", TOGGLES)
        self.assertIn('"GOLD_AI_TRENDLINE_ENABLED", False', TOGGLES)
        self.assertIn('"GOLD_AI_SETUP_TRENDLINE_BOUNCE_LONG"', TOGGLES)
        self.assertIn('"GOLD_AI_SETUP_TRENDLINE_BREAK_SHORT"', TOGGLES)

    def test_min_touches_default_three(self):
        self.assertIn('_env_int("GOLD_AI_TRENDLINE_MIN_TOUCHES", 3)', TRENDLINE)

    def test_info_level_line_diagnostics_present(self):
        self.assertIn("[gold-ai-trendline] validated side=", TRENDLINE)
        self.assertIn("[gold-ai-trendline] detected mode=", TRENDLINE)

    def test_session_playbook_includes_trendline_setups(self):
        self.assertIn('"trendline_bounce_long"', PLAYBOOK)
        self.assertIn('"trendline_break_short"', PLAYBOOK)

    def test_loop_sets_trendline_validator_profile(self):
        self.assertIn('decision["validator_profile"] = "trendline"', LOOP)
        self.assertIn('decision["trendline_level"]', LOOP)

    def test_validator_has_trendline_chase_guard(self):
        self.assertIn("TRENDLINE_BOUNCE_ENTRY_MAX_ATR", VALIDATOR)
        self.assertIn("TRENDLINE_BREAK_RETEST_ENTRY_MAX_ATR", VALIDATOR)
        self.assertIn("validator:entry_chasing_trendline", VALIDATOR)


if __name__ == "__main__":
    unittest.main()
