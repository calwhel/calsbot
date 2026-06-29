from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUTES = (ROOT / "app" / "gold_ai_trader" / "routes.py").read_text()


class GoldAiStatusReconcileSourceTests(unittest.TestCase):
    def test_status_endpoint_has_reconcile_throttle(self):
        self.assertIn("_STATUS_RECONCILE_LAST_RUN: Dict[int, float] = {}", ROUTES)
        self.assertIn("def _should_run_status_reconcile(user_id: int) -> bool:", ROUTES)
        self.assertIn("_STATUS_RECONCILE_INTERVAL_S", ROUTES)
        self.assertIn("_STATUS_RECONCILE_TIMEOUT_S", ROUTES)

    def test_status_endpoint_triggers_targeted_broker_reconcile(self):
        self.assertIn(
            "from app.services.strategy_executor import _reconcile_forex_closes",
            ROUTES,
        )
        self.assertIn("_reconcile_forex_closes(user_id=uid)", ROUTES)
        self.assertIn("await asyncio.wait_for(", ROUTES)
        self.assertIn("[gold-ai-trader] status bg reconcile timed out uid=%s", ROUTES)


if __name__ == "__main__":
    unittest.main()
