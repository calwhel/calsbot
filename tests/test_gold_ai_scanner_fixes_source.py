import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCANNER = (ROOT / "app/gold_ai_trader/scanner.py").read_text(encoding="utf-8")
TOGGLES = (ROOT / "app/gold_ai_trader/setup_toggles.py").read_text(encoding="utf-8")
SCORE = (ROOT / "app/gold_ai_trader/structure_score.py").read_text(encoding="utf-8")
ROUTING = (ROOT / "app/gold_ai_trader/entry_routing.py").read_text(encoding="utf-8")


class GoldAiScannerFixesSourceTests(unittest.TestCase):
    def test_detector_exceptions_log_warning_not_debug(self):
        self.assertIn("setup detector error setup=%s kind=%s: %s", SCANNER)
        self.assertIn("logger.warning(", SCANNER)
        self.assertNotIn('logger.debug("[gold-ai-trader] scan %s: %s", ctype, exc)', SCANNER)

    def test_fvg_bull_bear_removed_from_toggles_and_routing(self):
        self.assertNotIn('"fvg_bull":', TOGGLES)
        self.assertNotIn('"fvg_bear":', TOGGLES)
        self.assertNotIn("GOLD_AI_SETUP_FVG_BULL", TOGGLES)
        self.assertNotIn("GOLD_AI_SETUP_FVG_BEAR", TOGGLES)
        self.assertNotIn('"fvg_bull": 2', SCORE)
        self.assertNotIn('"fvg_bear": 2', SCORE)
        self.assertNotIn('"fvg_bull"', ROUTING)
        self.assertNotIn('"fvg_bear"', ROUTING)

    def test_asian_sweep_defaults_disabled(self):
        self.assertIn('"asian_sweep_bull": False', TOGGLES)
        self.assertIn('"asian_sweep_bear": False', TOGGLES)
        self.assertIn('"disp_bull": False', TOGGLES)
        self.assertIn('"liq_sweep_bull": False', TOGGLES)
        self.assertIn('"judas_bull": False', TOGGLES)
        self.assertIn('"judas_bear": False', TOGGLES)


if __name__ == "__main__":
    unittest.main()
