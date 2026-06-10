"""TP/SL sanity validation and price derivation for cTrader orders."""
import unittest

from app.services.ctrader_sltp import (
    compute_sltp_prices,
    relative_sltp_wire,
    validate_sltp_sanity,
)


class TestSltpSanity(unittest.TestCase):
    def test_short_sltp_correct_side(self):
        entry = 4160.03
        tp, sl = compute_sltp_prices("SHORT", entry, tp_pct=0.75, sl_pct=0.25)
        self.assertLess(tp, entry)
        self.assertGreater(sl, entry)
        self.assertTrue(validate_sltp_sanity("SHORT", entry, sl, tp))

    def test_long_sltp_correct_side(self):
        entry = 1.0850
        tp, sl = compute_sltp_prices("LONG", entry, tp_pct=0.5, sl_pct=0.25)
        self.assertGreater(tp, entry)
        self.assertLess(sl, entry)
        self.assertTrue(validate_sltp_sanity("LONG", entry, sl, tp))

    def test_rejects_short_with_sl_below_entry(self):
        entry = 4160.03
        self.assertFalse(validate_sltp_sanity("SHORT", entry, 4138.0, 4129.0))

    def test_relative_wire_uses_percentage_distance(self):
        entry = 4160.0
        rel_sl, rel_tp = relative_sltp_wire(entry, 0.25, 0.75)
        self.assertEqual(rel_sl, int(round(entry * 0.25 / 100 * 100_000)))
        self.assertEqual(rel_tp, int(round(entry * 0.75 / 100 * 100_000)))


if __name__ == "__main__":
    unittest.main()
