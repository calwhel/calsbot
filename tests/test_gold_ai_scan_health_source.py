from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "app" / "templates" / "gold_ai_trader.html").read_text()
ROUTES = (ROOT / "app" / "gold_ai_trader" / "routes.py").read_text()
STATE = (ROOT / "app" / "gold_ai_trader" / "state.py").read_text()
FUNNEL_PERSIST = (ROOT / "app" / "gold_ai_trader" / "funnel_persist.py").read_text()
LOOP = (ROOT / "app" / "gold_ai_trader" / "loop.py").read_text()


class GoldAiScanHealthSourceTests(unittest.TestCase):
    def test_template_has_explicit_scan_health_line_and_helpers(self):
        self.assertIn('id="scan-health-line"', TEMPLATE)
        self.assertIn("function deriveScanHealth(rt, cfg, sessionNow)", TEMPLATE)
        self.assertIn("rt._recent_events = (j.recent_funnel_events || []).slice(0, 10);", TEMPLATE)
        self.assertIn("Scanner live · heartbeat", TEMPLATE)
        self.assertIn("Scanner stale in open session", TEMPLATE)
        self.assertIn("Session open but scanner paused", TEMPLATE)
        self.assertIn("no shared heartbeat yet", TEMPLATE)

    def test_config_payload_exposes_scan_interval(self):
        self.assertIn('"scan_interval_s": cfg.scan_interval_s', ROUTES)

    def test_runtime_scan_timestamp_is_explicit_utc(self):
        self.assertIn('datetime.utcnow().replace(microsecond=0).isoformat() + "Z"', STATE)

    def test_scan_events_are_persisted_with_throttle(self):
        self.assertIn('"scan",', FUNNEL_PERSIST)
        self.assertIn("GOLD_AI_SCAN_HEARTBEAT_PERSIST_S", FUNNEL_PERSIST)
        self.assertIn("_last_scan_persist_mono", FUNNEL_PERSIST)

    def test_status_endpoint_kicks_background_loop(self):
        self.assertIn("from app.gold_ai_trader.loop import maybe_start_background_loop", ROUTES)
        self.assertIn("await maybe_start_background_loop()", ROUTES)

    def test_loop_reclaims_stale_lock_after_repeated_misses(self):
        self.assertIn("def _reclaim_stale_gold_ai_locks(*, min_idle_seconds: float) -> int:", LOOP)
        self.assertIn("terminate_lock_holders(", LOOP)
        self.assertIn("GOLD_AI_LOCK_RECLAIM_AFTER_MISSES", LOOP)
        self.assertIn("GOLD_AI_LOCK_RECLAIM_MIN_IDLE_S", LOOP)


if __name__ == "__main__":
    unittest.main()
