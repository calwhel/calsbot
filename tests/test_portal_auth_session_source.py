from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
PORTAL_AUTH = (ROOT / "app/portal_auth.py").read_text()
PORTAL_SESSION = (ROOT / "app/portal_session.py").read_text()
GOLD_ROUTES = (ROOT / "app/gold_ai_trader/routes.py").read_text()
PORTAL_SERVER = (ROOT / "strategy_portal_server.py").read_text()
GOLD_HTML = (ROOT / "app/templates/gold_ai_trader.html").read_text()
LOGIN_HTML = (ROOT / "app/templates/login.html").read_text()


class PortalAuthSessionSourceTests(unittest.TestCase):
    def test_portal_auth_exports_session_helpers(self):
        self.assertIn("def resolve_session_uid_with_user", PORTAL_AUTH)
        self.assertIn("def session_token_from_request", PORTAL_AUTH)
        self.assertIn("def stale_session_redirect", PORTAL_AUTH)

    def test_portal_session_warns_on_ephemeral_secret(self):
        self.assertIn("per-process session signer", PORTAL_SESSION)
        self.assertIn("def delete_session_cookie", PORTAL_SESSION)
        self.assertIn("COOKIE_NAME = ", PORTAL_SESSION)

    def test_gold_ai_page_requires_auth_and_no_cookie_overwrite(self):
        self.assertIn("_gold_ai_page_auth_redirect", GOLD_ROUTES)
        self.assertIn("login_redirect", GOLD_ROUTES)
        self.assertIn("no-cache, no-store, must-revalidate", GOLD_ROUTES)
        self.assertNotIn("set_session_cookie", GOLD_ROUTES)
        self.assertNotIn("make_session_token", GOLD_ROUTES)

    def test_gold_ai_ui_differentiates_auth_errors(self):
        self.assertIn("Admin access required", GOLD_HTML)
        self.assertIn("r.status === 401", GOLD_HTML)
        self.assertIn("r.status === 403", GOLD_HTML)

    def test_legacy_strategies_requires_login(self):
        self.assertIn("login_redirect", PORTAL_SERVER)
        self.assertIn('return login_redirect(f"/strategies?uid={norm_uid}")', PORTAL_SERVER)

    def test_logout_deletes_secure_cookie(self):
        self.assertIn("delete_session_cookie", PORTAL_SERVER)

    def test_login_page_handles_session_messages(self):
        self.assertIn("session_expired", LOGIN_HTML)
        self.assertIn("admin_required", LOGIN_HTML)


if __name__ == "__main__":
    unittest.main()
