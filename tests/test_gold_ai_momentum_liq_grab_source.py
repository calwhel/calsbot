import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = (ROOT / "app/gold_ai_trader/scanner.py").read_text(encoding="utf-8")
TOGGLES = (ROOT / "app/gold_ai_trader/setup_toggles.py").read_text(encoding="utf-8")
PLAYBOOK = (ROOT / "app/gold_ai_trader/session_playbook.py").read_text(encoding="utf-8")
ROUTING = (ROOT / "app/gold_ai_trader/entry_routing.py").read_text(encoding="utf-8")
SCORE = (ROOT / "app/gold_ai_trader/structure_score.py").read_text(encoding="utf-8")
VALIDATOR = (ROOT / "app/gold_ai_trader/decision_validator.py").read_text(encoding="utf-8")
LOOP = (ROOT / "app/gold_ai_trader/loop.py").read_text(encoding="utf-8")
MOMENTUM = (ROOT / "app/gold_ai_trader/momentum.py").read_text(encoding="utf-8")
LIQ_GRAB = (ROOT / "app/gold_ai_trader/liquidity_grab.py").read_text(encoding="utf-8")


class GoldAiMomentumLiquidityGrabSourceTests(unittest.TestCase):
    def test_six_new_setups_default_off(self):
        self.assertIn('"momentum_ema_bounce_long": False', TOGGLES)
        self.assertIn('"momentum_ema_bounce_short": False', TOGGLES)
        self.assertIn('"momentum_flag_break_long": False', TOGGLES)
        self.assertIn('"momentum_flag_break_short": False', TOGGLES)
        self.assertIn('"liquidity_grab_long": False', TOGGLES)
        self.assertIn('"liquidity_grab_short": False', TOGGLES)
        self.assertIn('return _env_bool("GOLD_AI_MOMENTUM_ENABLED", False)', TOGGLES)
        self.assertIn('return _env_bool("GOLD_AI_LIQ_GRAB_ENABLED", False)', TOGGLES)

    def test_scanner_wires_all_six_setups_and_detectors(self):
        self.assertIn('"momentum_ema_bounce_long"', SCANNER)
        self.assertIn('"momentum_ema_bounce_short"', SCANNER)
        self.assertIn('"momentum_flag_break_long"', SCANNER)
        self.assertIn('"momentum_flag_break_short"', SCANNER)
        self.assertIn('"liquidity_grab_long"', SCANNER)
        self.assertIn('"liquidity_grab_short"', SCANNER)
        self.assertIn("eval_momentum_ema_bounce(", SCANNER)
        self.assertIn("eval_momentum_flag_break(", SCANNER)
        self.assertIn("eval_liquidity_grab(", SCANNER)
        self.assertIn("if detector_meta:", SCANNER)
        self.assertIn("raw.update(detector_meta)", SCANNER)

    def test_playbook_routing_and_score_include_new_setups(self):
        self.assertIn('"momentum_ema_bounce_long"', PLAYBOOK)
        self.assertIn('"momentum_flag_break_short"', PLAYBOOK)
        self.assertIn('"liquidity_grab_long"', PLAYBOOK)
        self.assertIn('"momentum_ema_bounce_"', ROUTING)
        self.assertIn('"momentum_flag_break_"', ROUTING)
        self.assertIn('"liquidity_grab_"', ROUTING)
        self.assertIn('"momentum_ema_bounce_long": 6', SCORE)
        self.assertIn('"liquidity_grab_short": 7', SCORE)

    def test_validator_and_loop_have_breakout_anchor_profiles(self):
        self.assertIn("MOMENTUM_FLAG_ENTRY_MAX_ATR", VALIDATOR)
        self.assertIn("MOMENTUM_FLAG_RETEST_ENTRY_MAX_ATR", VALIDATOR)
        self.assertIn("LIQ_GRAB_ENTRY_MAX_ATR", VALIDATOR)
        self.assertIn("validator:entry_chasing_momentum_flag", VALIDATOR)
        self.assertIn("validator:entry_chasing_liquidity_grab", VALIDATOR)
        self.assertIn('decision["validator_profile"] = "momentum_flag"', LOOP)
        self.assertIn('decision["momentum_break_level"]', LOOP)
        self.assertIn('decision["validator_profile"] = "liquidity_grab"', LOOP)
        self.assertIn('decision["liq_grab_mss_level"]', LOOP)

    def test_info_diagnostics_exist_for_momentum_and_liq_grab(self):
        self.assertIn("[gold-ai-momentum] detected type=ema_bounce", MOMENTUM)
        self.assertIn("[gold-ai-momentum] detected type=flag_break", MOMENTUM)
        self.assertIn("[gold-ai-liquidity-grab] detected dir=", LIQ_GRAB)
        self.assertIn("sweep=", LIQ_GRAB)
        self.assertIn("mss=", LIQ_GRAB)

    def test_liquidity_grab_overlap_skip_and_mss_requirement_wired(self):
        self.assertIn('"PDH": pdh', LIQ_GRAB)
        self.assertIn('"PDL": pdl', LIQ_GRAB)
        self.assertIn('"ASIAN_HIGH": asian_hi', LIQ_GRAB)
        self.assertIn('"ASIAN_LOW": asian_lo', LIQ_GRAB)
        self.assertIn('"EQH": eqh', LIQ_GRAB)
        self.assertIn('"EQL": eql', LIQ_GRAB)
        self.assertIn("[gold-ai-liquidity-grab] overlap_skip", LIQ_GRAB)
        self.assertIn("if mss_idx is None:", LIQ_GRAB)
        self.assertIn("Liquidity grab {direction}: no sweep+MSS sequence", LIQ_GRAB)


if __name__ == "__main__":
    unittest.main()
