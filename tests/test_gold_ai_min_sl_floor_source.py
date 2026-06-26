import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = (ROOT / "app/gold_ai_trader/decision_validator.py").read_text(encoding="utf-8")
CLAUDE = (ROOT / "app/gold_ai_trader/claude.py").read_text(encoding="utf-8")


class GoldAiMinSlFloorSourceTests(unittest.TestCase):
    def test_validator_has_min_sl_pips_floor(self):
        self.assertIn('MIN_SL_PIPS = _env_float("GOLD_AI_MIN_SL_PIPS", 60.0)', VALIDATOR)
        self.assertIn("validator:sl_too_tight", VALIDATOR)
        self.assertIn("platform_pips_from_price_delta", VALIDATOR)

    def test_prompts_include_min_stop_instruction(self):
        self.assertIn("minimum stop distance of 60 platform pips", CLAUDE)
        self.assertIn("Minimum stop distance is 60 platform pips", CLAUDE)


if __name__ == "__main__":
    unittest.main()
