import unittest
from datetime import datetime

from app.services.session_filter import SESSIONS, is_in_allowed_session


class TestSessionFilter(unittest.TestCase):
    def test_disabled_always_allows(self):
        ok, reason = is_in_allowed_session({}, datetime(2026, 6, 10, 3, 0))
        self.assertTrue(ok)
        self.assertEqual(reason, "")

    def test_london_window(self):
        cfg = {"sessions_enabled": True, "allowed_sessions": ["london"]}
        ok, _ = is_in_allowed_session(cfg, datetime(2026, 6, 10, 10, 0))
        self.assertTrue(ok)
        ok, reason = is_in_allowed_session(cfg, datetime(2026, 6, 10, 20, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "session_filter")

    def test_sydney_crosses_midnight(self):
        cfg = {"sessions_enabled": True, "allowed_sessions": ["sydney"]}
        ok, _ = is_in_allowed_session(cfg, datetime(2026, 6, 10, 22, 0))
        self.assertTrue(ok)
        ok, _ = is_in_allowed_session(cfg, datetime(2026, 6, 10, 3, 0))
        self.assertTrue(ok)
        ok, reason = is_in_allowed_session(cfg, datetime(2026, 6, 10, 12, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "session_filter")

    def test_custom_window(self):
        cfg = {
            "sessions_enabled": True,
            "allowed_sessions": [],
            "session_custom": {"start": "08:00", "end": "09:00"},
        }
        ok, _ = is_in_allowed_session(cfg, datetime(2026, 6, 10, 8, 30))
        self.assertTrue(ok)
        ok, reason = is_in_allowed_session(cfg, datetime(2026, 6, 10, 10, 0))
        self.assertFalse(ok)
        self.assertEqual(reason, "session_filter")

    def test_sessions_dict_has_expected_keys(self):
        self.assertIn("london", SESSIONS)
        self.assertIn("newyork", SESSIONS)


if __name__ == "__main__":
    unittest.main()
