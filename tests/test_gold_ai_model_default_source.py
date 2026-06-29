from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (ROOT / "app" / "gold_ai_trader" / "config.py").read_text()
MODELS = (ROOT / "app" / "gold_ai_trader" / "models.py").read_text()
SCHEMA = (ROOT / "app" / "gold_ai_trader" / "schema.py").read_text()
ROUTING = (ROOT / "app" / "gold_ai_trader" / "entry_routing.py").read_text()
FIRE = (ROOT / "app" / "gold_ai_trader" / "fire_time_validation.py").read_text()


class GoldAiModelDefaultSourceTests(unittest.TestCase):
    def test_sonnet_is_default_in_runtime_and_db_model(self):
        self.assertIn('model="claude-sonnet-4-6"', CONFIG)
        self.assertIn(
            'model=os.environ.get("GOLD_AI_TRADER_MODEL", "claude-sonnet-4-6")',
            CONFIG,
        )
        self.assertIn(
            'model = Column(String(64), default="claude-sonnet-4-6", nullable=False)',
            MODELS,
        )

    def test_seed_upgrades_haiku_to_sonnet(self):
        self.assertIn('== "claude-haiku-4-5"', SCHEMA)
        self.assertIn('row.model = "claude-sonnet-4-6"', SCHEMA)


class GoldAiFastExecSourceTests(unittest.TestCase):
    def test_ob_market_entry_default(self):
        self.assertIn("GOLD_AI_OB_MARKET_ENTRY", ROUTING)
        self.assertIn("ob_market_entry_enabled", ROUTING)

    def test_fire_time_revalidate_module(self):
        self.assertIn("revalidate_before_fire", FIRE)
        self.assertIn("GOLD_AI_FIRE_TIME_REVALIDATE", FIRE)
        self.assertIn("refresh_spot_after_claude", FIRE)


if __name__ == "__main__":
    unittest.main()
