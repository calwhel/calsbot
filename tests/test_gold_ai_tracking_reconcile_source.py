import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FX_EVENTS = (ROOT / "app/services/ctrader_execution_events.py").read_text(encoding="utf-8")
FX_EXECUTOR = (ROOT / "app/services/strategy_executor.py").read_text(encoding="utf-8")
CTRADER = (ROOT / "app/services/ctrader_client.py").read_text(encoding="utf-8")
GOLD_EXECUTOR = (ROOT / "app/gold_ai_trader/executor.py").read_text(encoding="utf-8")
STRATEGY_HEAL = (ROOT / "app/services/strategy_heal.py").read_text(encoding="utf-8")


class GoldAiTrackingReconcileSourceTests(unittest.TestCase):
    def test_event_path_has_order_id_fallback_matching(self):
        self.assertIn("match_basis", FX_EVENTS)
        self.assertIn("order_id=order_id", FX_EVENTS)
        self.assertIn("ctrader-execution-event-order-fallback", FX_EVENTS)

    def test_reconcile_path_has_order_close_recovery(self):
        self.assertIn("get_order_close_detail_for_user", FX_EXECUTOR)
        self.assertIn("_FX_RECONCILE_ORDERLESS_ESTIMATE_AFTER_S", FX_EXECUTOR)
        self.assertIn("ctrader-reconcile-broker-order", FX_EXECUTOR)
        self.assertIn("ctrader-reconcile-estimated-orderless", FX_EXECUTOR)

    def test_ctrader_client_supports_order_close_lookup(self):
        self.assertIn("def get_order_close_detail_for_user(", CTRADER)
        self.assertIn("_position_id_from_order_deals(", CTRADER)
        self.assertIn("deal_list_window_order_id", CTRADER)

    def test_gold_executor_persists_order_and_position_tokens(self):
        self.assertIn("ord={order_id}", GOLD_EXECUTOR)
        self.assertIn("pos={position_id}", GOLD_EXECUTOR)

    def test_untracked_heal_checks_position_column_too(self):
        self.assertIn('getattr(execution, "ctrader_position_id"', STRATEGY_HEAL)
        self.assertIn("_execution_lacks_position_id(ex)", STRATEGY_HEAL)


if __name__ == "__main__":
    unittest.main()
