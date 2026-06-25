from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUTES = (ROOT / "app" / "gold_ai_trader" / "routes.py").read_text()
TEMPLATE = (ROOT / "app" / "templates" / "gold_ai_trader.html").read_text()


class GoldAiCostStatsSourceTests(unittest.TestCase):
    def test_stats_today_loads_cost_independently(self):
        self.assertIn("def _load_stat(name: str, loader, *, round_digits: Optional[int] = None) -> None:", ROUTES)
        self.assertIn('_load_stat("cost_usd", lambda: cost_today_usd(db), round_digits=6)', ROUTES)
        self.assertIn("stats_today %s failed", ROUTES)

    def test_dashboard_cost_display_keeps_sub_dollar_precision(self):
        self.assertIn("if (cost > 0 && cost < 0.0001)", TEMPLATE)
        self.assertIn("cost.toFixed(4)", TEMPLATE)
        self.assertIn("Reset Claude usage counters (calls + API cost since reset) to 0?", TEMPLATE)


if __name__ == "__main__":
    unittest.main()
