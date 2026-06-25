from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
LEARNING = (ROOT / "app" / "gold_ai_trader" / "learning.py").read_text()


class GoldAiOutcomeLoggingSourceTests(unittest.TestCase):
    def test_record_outcome_logs_decision_exec_result_and_r_multiple(self):
        self.assertIn("[gold-ai-trader] outcome recorded decision_id=%s exec_id=%s", LEARNING)
        self.assertIn("result=%s pnl_pct=%s r_multiple=%s", LEARNING)
        self.assertIn("round(float(r_mult), 4) if r_mult is not None else None", LEARNING)


if __name__ == "__main__":
    unittest.main()
