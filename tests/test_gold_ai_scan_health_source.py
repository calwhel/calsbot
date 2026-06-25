from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "app" / "templates" / "gold_ai_trader.html").read_text()
ROUTES = (ROOT / "app" / "gold_ai_trader" / "routes.py").read_text()
STATE = (ROOT / "app" / "gold_ai_trader" / "state.py").read_text()


class GoldAiScanHealthSourceTests(unittest.TestCase):
    def test_template_has_explicit_scan_health_line_and_helpers(self):
        self.assertIn('id="scan-health-line"', TEMPLATE)
        self.assertIn("function deriveScanHealth(rt, cfg, sessionNow)", TEMPLATE)
        self.assertIn("Scanner live · heartbeat", TEMPLATE)
        self.assertIn("Scanner stale in open session", TEMPLATE)
        self.assertIn("Session open but scanner paused", TEMPLATE)

    def test_config_payload_exposes_scan_interval(self):
        self.assertIn('"scan_interval_s": cfg.scan_interval_s', ROUTES)

    def test_runtime_scan_timestamp_is_explicit_utc(self):
        self.assertIn('datetime.utcnow().replace(microsecond=0).isoformat() + "Z"', STATE)


if __name__ == "__main__":
    unittest.main()
