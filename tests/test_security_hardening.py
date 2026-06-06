from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
PORTAL = (ROOT / "strategy_portal_server.py").read_text()
CONFIG = (ROOT / "app" / "config.py").read_text()
TWITTER_POSTER = (ROOT / "app" / "services" / "twitter_poster.py").read_text()
TRADE_TRACKER = (ROOT / "app" / "services" / "trade_tracker.py").read_text()


class SecurityHardeningSourceTests(unittest.TestCase):
    def test_no_hardcoded_admin_or_session_secret_fallbacks(self):
        combined = "\n".join([PORTAL, CONFIG, TWITTER_POSTER, TRADE_TRACKER])

        self.assertNotIn("tradehub-portal-secret-2025", combined)
        self.assertNotRegex(combined, r'ADMIN_SECRET"\s*,\s*"5603353066"')
        self.assertNotRegex(combined, r'OWNER_TELEGRAM_ID"\s*,\s*"5603353066"')
        self.assertNotRegex(combined, r"ADMIN_CHAT_ID\s*=\s*5603353066")

    def test_uid_api_guard_requires_bound_session_token(self):
        self.assertIn("async def _require_bound_uid_for_api", PORTAL)
        self.assertIn("_get_session_uid(request)", PORTAL)
        self.assertIn("_get_request_token_uid(request)", PORTAL)
        self.assertIn("A signed session token is required for this UID.", PORTAL)
        self.assertIn("ALLOW_LEGACY_UID_API_AUTH", PORTAL)

    def test_mobile_uid_only_login_disabled_by_default(self):
        self.assertIn("UID-only mobile login is disabled", PORTAL)
        self.assertIn('"session_token": session_token', PORTAL)
        self.assertIn('"auth_token":    session_token', PORTAL)

    def test_bitunix_keys_are_encrypted_on_portal_save(self):
        self.assertIn("from app.utils.encryption import encrypt_api_key", PORTAL)
        self.assertIn("prefs.bitunix_api_key = encrypt_api_key(api_key)", PORTAL)
        self.assertIn("prefs.bitunix_api_secret = encrypt_api_key(api_secret)", PORTAL)
        self.assertNotIn("prefs.bitunix_api_key = api_key", PORTAL)
        self.assertNotIn("prefs.bitunix_api_secret = api_secret", PORTAL)

    def test_oxapay_webhook_fails_closed(self):
        self.assertIn("Missing HMAC signature", PORTAL)
        self.assertIn("Payment status could not be verified", PORTAL)
        self.assertIn("Payment amount or currency mismatch", PORTAL)
        self.assertRegex(PORTAL, re.compile(r"if not merchant_key:.*?raise HTTPException", re.S))
        self.assertRegex(PORTAL, re.compile(r"if not sig:.*?raise HTTPException", re.S))
        self.assertIn("oxapay.check_payment_status", PORTAL)


if __name__ == "__main__":
    unittest.main()
