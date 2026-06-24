from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ROUTES = (ROOT / "app" / "gold_ai_trader" / "routes.py").read_text()
TEMPLATE = (ROOT / "app" / "templates" / "gold_ai_trader.html").read_text()


class GoldAiAsiaReadinessSourceTests(unittest.TestCase):
    def test_default_admin_uid_set_includes_operator_uid(self):
        self.assertIn('uids = {"TH-YP0BADA8", "TH-ZKJO6YKX"}', ROUTES)
        self.assertIn("u.uid in _gold_ai_admin_uids()", ROUTES)

    def test_timeline_and_copy_include_asia_session(self):
        self.assertIn('id="block-asia"', TEMPLATE)
        self.assertIn("Asia, London and New York session windows", TEMPLATE)
        self.assertIn("Outside Asia, London, and New York session windows", TEMPLATE)
        self.assertIn("ASIA ${asiaS}–${asiaE}", TEMPLATE)


if __name__ == "__main__":
    unittest.main()
