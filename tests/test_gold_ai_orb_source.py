import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "app/gold_ai_trader/config.py").read_text(encoding="utf-8")
MODELS = (ROOT / "app/gold_ai_trader/models.py").read_text(encoding="utf-8")
SCHEMA = (ROOT / "app/gold_ai_trader/schema.py").read_text(encoding="utf-8")
CLAUDE = (ROOT / "app/gold_ai_trader/claude.py").read_text(encoding="utf-8")
ORB = (ROOT / "app/gold_ai_trader/orb.py").read_text(encoding="utf-8")
LOOP = (ROOT / "app/gold_ai_trader/loop.py").read_text(encoding="utf-8")
GUARDRAILS = (ROOT / "app/gold_ai_trader/guardrails.py").read_text(encoding="utf-8")
VALIDATOR = (ROOT / "app/gold_ai_trader/decision_validator.py").read_text(encoding="utf-8")


class GoldAiOrbSourceTests(unittest.TestCase):
    def test_config_has_orb_env_toggles_and_defaults(self):
        self.assertIn("orb_enabled: bool", CONFIG)
        self.assertIn("orb_max_calls_day: int", CONFIG)
        self.assertIn("orb_confidence_threshold: int", CONFIG)
        self.assertIn("GOLD_AI_ORB_ENABLED", CONFIG)
        self.assertIn("GOLD_AI_ORB_RANGE_MINUTES", CONFIG)
        self.assertIn("GOLD_AI_ORB_MAX_CALLS_DAY", CONFIG)
        self.assertIn("orb_enabled=False", CONFIG)

    def test_orb_state_model_and_schema_registration(self):
        self.assertIn("class GoldAiOrbState(Base):", MODELS)
        self.assertIn('__tablename__ = "gold_ai_orb_state"', MODELS)
        self.assertIn("uq_gold_ai_orb_state_day_session", MODELS)
        self.assertIn("GoldAiOrbState.__table__", SCHEMA)

    def test_dedicated_orb_prompt_and_decision_path(self):
        self.assertIn("def orb_system_prompt(", CLAUDE)
        self.assertIn("async def decide_orb(", CLAUDE)
        self.assertIn("Opening Range Breakout", CLAUDE)

    def test_orb_detector_and_loop_integration(self):
        self.assertIn("async def detect_orb_signal(", ORB)
        self.assertIn("build_orb_context(", ORB)
        self.assertIn("setup detector error setup=%s kind=%s: %s", LOOP)
        self.assertIn("check_can_call_orb", LOOP)
        self.assertIn("candidate_type=signal.setup_type", LOOP)

    def test_orb_call_budget_and_validator_profile(self):
        self.assertIn("def check_can_call_orb(", GUARDRAILS)
        self.assertIn("orb_reserve_global_calls", GUARDRAILS)
        self.assertIn("validator_profile", VALIDATOR)
        self.assertIn("entry_chasing_orb", VALIDATOR)


if __name__ == "__main__":
    unittest.main()
