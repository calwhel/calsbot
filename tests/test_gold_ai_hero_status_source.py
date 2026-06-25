from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "app" / "templates" / "gold_ai_trader.html").read_text()


class GoldAiHeroStatusSourceTests(unittest.TestCase):
    def test_hero_uses_utc_session_window_fallback(self):
        self.assertIn("function sessionNowUtc(cfg, sharedHours)", TEMPLATE)
        self.assertIn("const sessionNow = sessionNowUtc(cfg, j.shared_session_hours);", TEMPLATE)
        self.assertIn("if (sessionNow) {", TEMPLATE)
        self.assertIn("Session is open — waiting for next scan cycle", TEMPLATE)

    def test_hero_surfaces_non_session_pause_reasons(self):
        self.assertIn("if (status === 'outside_killzone')", TEMPLATE)
        self.assertIn("if (status.startsWith('news:'))", TEMPLATE)
        self.assertIn("if (status.startsWith('data_quality:'))", TEMPLATE)
        self.assertIn("if (status === 'max_calls_day')", TEMPLATE)

    def test_hero_in_trade_uses_latest_executed_take_outcome(self):
        self.assertIn("const latestExecutedTake = feed.find(d =>", TEMPLATE)
        self.assertIn("String(latestExecutedTake.execution_outcome || '').toUpperCase() === 'OPEN'", TEMPLATE)
        self.assertIn("const openTrade = latestExecutedTake", TEMPLATE)

    def test_session_now_prefers_new_york_over_london_in_overlap(self):
        ny_idx = TEMPLATE.find("if (nowH >= nyS && nowH < nyE) return 'new_york';")
        lon_idx = TEMPLATE.find("if (nowH >= lonS && nowH < lonE) return 'london';")
        self.assertGreaterEqual(ny_idx, 0)
        self.assertGreaterEqual(lon_idx, 0)
        self.assertLess(ny_idx, lon_idx)


if __name__ == "__main__":
    unittest.main()
