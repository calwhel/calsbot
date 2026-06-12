"""Platform vs broker pip unit conversions."""
import unittest

from app.services.pip_units import (
    broker_pip_size_from_metadata,
    broker_pips_to_platform_pips,
    format_platform_pip_move,
    platform_pip_size,
    platform_pips_from_price_delta,
    platform_pips_to_broker_pips,
    sl_pips_platform,
    to_broker_relative_wire_units,
)


class TestPlatformPipSize(unittest.TestCase):
    def test_xauusd_platform_pip_is_point_one(self):
        self.assertEqual(platform_pip_size("XAUUSD"), 0.10)

    def test_eurusd_platform_pip(self):
        self.assertEqual(platform_pip_size("EURUSD"), 0.0001)

    def test_broker_xau_pip_position_differs(self):
        brok = broker_pip_size_from_metadata(pip_position=2, symbol="XAUUSD")
        self.assertEqual(brok, 0.01)
        self.assertEqual(platform_pips_to_broker_pips("XAUUSD", 15.0, pip_position=2), 150.0)
        self.assertAlmostEqual(
            broker_pips_to_platform_pips("XAUUSD", 68.5, pip_position=2),
            6.85,
            places=2,
        )


class TestStaleGuardPipMath(unittest.TestCase):
    def test_gold_flutter_not_inflated_tenfold(self):
        # $0.685 move = 6.85 platform pips at 0.10 — NOT 68.5 broker ticks
        pips = platform_pips_from_price_delta("XAUUSD", 0.685)
        self.assertAlmostEqual(pips, 6.85, places=1)
        msg = format_platform_pip_move("XAUUSD", 2650.0, 2650.685)
        self.assertIn("pip_size=0.1", msg)
        self.assertIn("6.8 pips", msg)

    def test_sl_pips_platform_uses_platform_units(self):
        sl_p = sl_pips_platform("XAUUSD", 2650.0, 2647.5)
        self.assertAlmostEqual(sl_p, 25.0, places=1)


class TestBrokerWireUnits(unittest.TestCase):
    def test_relative_wire_matches_pct_distance(self):
        entry = 4160.0
        wire = to_broker_relative_wire_units(entry * 0.0025)
        self.assertEqual(wire, int(round(entry * 0.0025 * 100_000)))


if __name__ == "__main__":
    unittest.main()
