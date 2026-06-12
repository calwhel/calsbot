"""Tests for GET /api/ctrader/health."""
import unittest
from unittest.mock import MagicMock, patch

try:
    from fastapi.testclient import TestClient
    import strategy_portal_server as sps
    _HAS_APP = True
except Exception:
    _HAS_APP = False


@unittest.skipUnless(_HAS_APP, "strategy_portal_server unavailable")
class TestCtraderHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(sps.app)

    @patch("strategy_portal_server._get_user_by_uid")
    @patch("app.services.ctrader_price_feed.feed_status")
    @patch("app.services.ctrader_client.audit_ctrader_credentials")
    def test_health_linked_feed_alive(self, mock_audit, mock_feed, mock_user):
        user = MagicMock()
        user.id = 42
        mock_user.return_value = user
        prefs = MagicMock()
        prefs.ctrader_access_token = "tok"
        prefs.ctrader_refresh_token = "rtok"
        prefs.ctrader_account_id = "123"
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = prefs

        with patch("app.database.SessionLocal", return_value=db):
            mock_audit.return_value = {"ok": True}
            mock_feed.return_value = {"last_tick_age_s": 12.0, "live": True, "symbol_count": 5}
            r = self.client.get("/api/ctrader/health?uid=TESTUID")

        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["linked"])
        self.assertTrue(body["token_valid"])
        self.assertTrue(body["feed_alive"])
        self.assertIn("last_check_ts", body)

    @patch("strategy_portal_server._get_user_by_uid")
    @patch("app.services.ctrader_price_feed.feed_status")
    @patch("app.services.ctrader_client.audit_ctrader_credentials")
    def test_health_token_invalid(self, mock_audit, mock_feed, mock_user):
        user = MagicMock()
        user.id = 7
        mock_user.return_value = user
        prefs = MagicMock()
        prefs.ctrader_access_token = "bad"
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = prefs

        with patch("app.database.SessionLocal", return_value=db):
            mock_audit.return_value = {"ok": False, "reason": "refresh chain denied"}
            mock_feed.return_value = {"last_tick_age_s": 5.0}
            r = self.client.get("/api/ctrader/health?uid=ABC")

        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["linked"])
        self.assertFalse(body["token_valid"])
