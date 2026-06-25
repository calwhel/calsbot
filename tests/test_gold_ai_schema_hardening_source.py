from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "app" / "gold_ai_trader" / "schema.py").read_text()
MOUNT = (ROOT / "app" / "gold_ai_trader" / "portal_mount.py").read_text()


class GoldAiSchemaHardeningSourceTests(unittest.TestCase):
    def test_schema_uses_db_retry_for_create_alter_and_column_scan(self):
        self.assertIn("from app.db_resilience import run_with_db_retry", SCHEMA)
        self.assertIn("label=\"gold-ai-schema-create-all\"", SCHEMA)
        self.assertIn("label=\"gold-ai-schema-missing-columns\"", SCHEMA)
        self.assertIn("label=f\"gold-ai-schema-alter:{table}.{column}\"", SCHEMA)

    def test_schema_retry_knobs_are_env_configurable(self):
        self.assertIn("GOLD_AI_SCHEMA_DDL_RETRY_ATTEMPTS", SCHEMA)
        self.assertIn("GOLD_AI_SCHEMA_DDL_RETRY_DELAY_S", SCHEMA)

    def test_startup_hard_fail_toggle_exists(self):
        self.assertIn("GOLD_AI_SCHEMA_STARTUP_HARD_FAIL", MOUNT)
        self.assertIn("schema startup hard-fail enabled", MOUNT)


if __name__ == "__main__":
    unittest.main()
