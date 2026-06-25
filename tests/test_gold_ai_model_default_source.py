from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "app" / "gold_ai_trader" / "config.py").read_text()
MODELS = (ROOT / "app" / "gold_ai_trader" / "models.py").read_text()
SCHEMA = (ROOT / "app" / "gold_ai_trader" / "schema.py").read_text()


class GoldAiModelDefaultSourceTests(unittest.TestCase):
    def test_opus_is_default_in_runtime_and_db_model(self):
        self.assertIn('model="claude-opus-4-8"', CONFIG)
        self.assertIn('model=os.environ.get("GOLD_AI_TRADER_MODEL", "claude-opus-4-8")', CONFIG)
        self.assertIn('model = Column(String(64), default="claude-opus-4-8", nullable=False)', MODELS)

    def test_seed_upgrades_haiku_to_opus(self):
        self.assertIn('== "claude-haiku-4-5"', SCHEMA)
        self.assertIn('row.model = "claude-opus-4-8"', SCHEMA)


if __name__ == "__main__":
    unittest.main()
