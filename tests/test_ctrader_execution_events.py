"""cTrader execution-event close classification."""
import unittest

from app.services.ctrader_execution_events import _classify_outcome


class TestCtraderExecutionEvents(unittest.TestCase):
    def test_win_on_positive_gross(self):
        self.assertEqual(_classify_outcome(100, 1.10, 1.12, "LONG"), "WIN")

    def test_loss_on_negative_gross(self):
        self.assertEqual(_classify_outcome(-50, 1.10, 1.08, "LONG"), "LOSS")

    def test_breakeven_near_entry(self):
        self.assertEqual(
            _classify_outcome(0, 2650.0, 2650.0, "LONG"),
            "BREAKEVEN",
        )


if __name__ == "__main__":
    unittest.main()
