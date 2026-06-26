import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "app/gold_ai_trader/config.py").read_text(encoding="utf-8")
CONTEXT = (ROOT / "app/gold_ai_trader/context.py").read_text(encoding="utf-8")
CLAUDE = (ROOT / "app/gold_ai_trader/claude.py").read_text(encoding="utf-8")
GUARDRAILS = (ROOT / "app/gold_ai_trader/guardrails.py").read_text(encoding="utf-8")


class GoldAiFreshDayContextSourceTests(unittest.TestCase):
    def test_config_exposes_history_toggle_default_off(self):
        self.assertIn("include_history_in_decisions: bool", CONFIG)
        self.assertIn("include_history_in_decisions=False", CONFIG)
        self.assertIn("GOLD_AI_INCLUDE_HISTORY_IN_DECISIONS", CONFIG)

    def test_guardrails_merge_carries_history_toggle(self):
        self.assertIn("include_history_in_decisions=env.include_history_in_decisions", GUARDRAILS)

    def test_context_suppresses_history_when_toggle_off(self):
        self.assertIn("if cfg.include_history_in_decisions:", CONTEXT)
        self.assertIn("Fresh-day mode: historical setup win-rate stats suppressed.", CONTEXT)
        self.assertIn("Fresh-day mode: prior decision/trade history is not used in this prompt.", CONTEXT)
        self.assertIn("Fresh-day mode: historical lessons disabled; focus on live setup quality.", CONTEXT)

    def test_prompt_tells_model_to_prioritize_live_structure(self):
        self.assertIn("If lessons/history are provided, treat them as secondary context only.", CLAUDE)
        self.assertIn("Do not treat tiny samples as predictive; prioritize current structure", CLAUDE)


if __name__ == "__main__":
    unittest.main()
