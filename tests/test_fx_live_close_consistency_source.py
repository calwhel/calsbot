from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
EXECUTOR = (ROOT / "app" / "services" / "strategy_executor.py").read_text()


class FxLiveCloseConsistencySourceTests(unittest.TestCase):
    def test_live_close_classifies_before_db_update(self):
        self.assertIn("outcome, label = close_classifier(", EXECUTOR)
        self.assertIn(
            'ex, float(exit_price), proposed_outcome=(outcome or "LOSS")',
            EXECUTOR,
        )
        self.assertRegex(
            EXECUTOR,
            r"outcome, label = close_classifier\([\s\S]*?UPDATE strategy_executions[\s\S]*?SET outcome=:o",
        )

    def test_estimated_close_uses_neutral_label(self):
        self.assertIn('if "estimated" in (source or "").lower():', EXECUTOR)
        self.assertIn('label = "market close (estimated)"', EXECUTOR)
        self.assertIn('"EST"', EXECUTOR)


if __name__ == "__main__":
    unittest.main()
