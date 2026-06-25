from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUTES = (ROOT / "app" / "gold_ai_trader" / "routes.py").read_text()
TEMPLATE = (ROOT / "app" / "templates" / "gold_ai_trader.html").read_text()


class GoldAiFeedOutcomeSourceTests(unittest.TestCase):
    def test_decision_feed_includes_execution_outcome_fields(self):
        self.assertIn('"execution_outcome": (execution_meta or {}).get("outcome")', ROUTES)
        self.assertIn('"execution_closed_ts": (execution_meta or {}).get("closed_ts")', ROUTES)

    def test_dashboard_open_badge_uses_execution_outcome(self):
        self.assertIn("const execOutcome = String(d.execution_outcome || '').toUpperCase();", TEMPLATE)
        self.assertIn("if (!execOutcome || execOutcome === 'OPEN')", TEMPLATE)
        self.assertIn("else if (execOutcome === 'LOSS')", TEMPLATE)
        self.assertIn("demo ' + (execOutcome ? execOutcome.toLowerCase() : 'order')", TEMPLATE)

    def test_hero_in_trade_requires_open_execution_outcome(self):
        self.assertIn("const latestExecutedTake = feed.find(d =>", TEMPLATE)
        self.assertIn("String(latestExecutedTake.execution_outcome || '').toUpperCase() === 'OPEN'", TEMPLATE)


if __name__ == "__main__":
    unittest.main()
