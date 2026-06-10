"""Trade management config + SL helpers."""
import unittest

from app.services.trade_management import (
    active_sl,
    effective_trade_mgmt_cfg,
    _validate_sl,
)


class TestTradeManagement(unittest.TestCase):
    def test_defaults_off_for_empty_config(self):
        cfg = effective_trade_mgmt_cfg({})
        self.assertFalse(cfg["breakeven_enabled"])
        self.assertFalse(cfg["partial_tp_enabled"])

    def test_legacy_breakeven_mapping(self):
        cfg = effective_trade_mgmt_cfg({"exit": {"breakeven_at_pips": 25}})
        self.assertTrue(cfg["breakeven_enabled"])
        self.assertEqual(cfg["breakeven_trigger_pips"], 25.0)

    def test_active_sl_prefers_current(self):
        class Ex:
            sl_price = 1.08
            current_sl = 1.085

        self.assertEqual(active_sl(Ex()), 1.085)

    def test_validate_sl_long(self):
        self.assertTrue(_validate_sl("LONG", 1.08, 1.09))
        self.assertFalse(_validate_sl("LONG", 1.10, 1.09))


if __name__ == "__main__":
    unittest.main()
