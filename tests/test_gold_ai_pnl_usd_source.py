from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
GUARDRAILS = (ROOT / "app" / "gold_ai_trader" / "guardrails.py").read_text()
EXECUTOR = (ROOT / "app" / "services" / "strategy_executor.py").read_text()
GOLD_EXEC = (ROOT / "app" / "gold_ai_trader" / "executor.py").read_text()
EVENTS = (ROOT / "app" / "services" / "ctrader_execution_events.py").read_text()
CTRADER_CLIENT = (ROOT / "app" / "services" / "ctrader_client.py").read_text()
TRADE_MGMT = (ROOT / "app" / "services" / "trade_management.py").read_text()


class GoldAiPnlUsdSourceTests(unittest.TestCase):
    def test_guardrails_uses_volume_price_fallback_before_rough_proxy(self):
        self.assertIn("broker_volume_units", GUARDRAILS)
        self.assertIn("move = float(ex.exit_price) - float(ex.entry_price)", GUARDRAILS)
        self.assertIn("total += move * units * sign", GUARDRAILS)

    def test_live_close_path_accepts_and_persists_pnl_usd(self):
        self.assertIn("pnl_usd: Optional[float] = None", EXECUTOR)
        self.assertIn("pnl_usd=:pu", EXECUTOR)
        self.assertIn('"pu": pnl_usd', EXECUTOR)
        self.assertIn("pnl_usd=float(pnl_usd) if pnl_usd is not None else None", EXECUTOR)

    def test_execution_events_pass_broker_gross_to_close(self):
        self.assertIn("pnl_usd=round(float(gross) / 100.0, 2)", EVENTS)
        self.assertIn('"gross_profit_usd": round(float(gross) / 100.0, 2)', CTRADER_CLIENT)

    def test_reconcile_audit_backfills_pnl_usd(self):
        self.assertIn('"pnl_usd_backfilled": 0', TRADE_MGMT)
        self.assertIn("stats[\"pnl_usd_backfilled\"] += 1", TRADE_MGMT)

    def test_gold_ai_execution_stores_broker_volume_units(self):
        self.assertIn("broker_volume_units=broker_units_i", GOLD_EXEC)
        self.assertIn("note += f\" | vol={broker_units_i}\"", GOLD_EXEC)


if __name__ == "__main__":
    unittest.main()
