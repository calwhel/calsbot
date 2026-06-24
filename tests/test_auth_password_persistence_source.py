from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PORTAL = (ROOT / "strategy_portal_server.py").read_text()


class AuthPasswordPersistenceSourceTests(unittest.TestCase):
    def test_write_safe_uid_lookup_helper_exists(self):
        self.assertIn("def _get_user_row_by_uid(uid: str, db: Session):", PORTAL)
        self.assertIn(
            "return db.query(User).filter(User.uid == uid).first()",
            PORTAL,
        )

    def test_uid_set_password_uses_write_safe_lookup_and_invalidates_cache(self):
        self.assertRegex(
            PORTAL,
            re.compile(
                r"def _set_pw\(\):.*?_get_user_row_by_uid\(uid, db\).*?"
                r"u\.password_hash = _hash_password\(password\).*?"
                r"db\.commit\(\).*?_invalidate_user_cache\(uid\)",
                re.S,
            ),
        )

    def test_settings_password_change_uses_write_safe_lookup_and_invalidates_cache(self):
        self.assertRegex(
            PORTAL,
            re.compile(
                r"@app\.put\(\"/api/settings/password\"\).*?def _do\(\):.*?"
                r"_get_user_row_by_uid\(uid, db\).*?"
                r"u\.password_hash = _hash_password\(new_password\).*?"
                r"db\.commit\(\).*?_invalidate_user_cache\(uid\)",
                re.S,
            ),
        )


if __name__ == "__main__":
    unittest.main()
