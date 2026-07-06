"""cTrader OAuth single-flight refresh — deploy-survival guards."""
import inspect
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import unittest

os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost/test")


class TestCtraderTokenRefresh(unittest.TestCase):
    def test_refresh_lock_id_in_lock_ids(self):
        from app.lock_ids import CTRADER_TOKEN_REFRESH_LOCK_NS

        self.assertEqual(CTRADER_TOKEN_REFRESH_LOCK_NS, 708_110_021)

    def test_startup_log_helpers(self):
        from app.services.ctrader_client import (
            _log_ctrader_token_startup,
            refresh_user_ctrader_token,
        )

        self.assertTrue(callable(_log_ctrader_token_startup))
        src = inspect.getsource(refresh_user_ctrader_token)
        self.assertIn("async with _token_refresh_lock", src)
        lock_src = inspect.getsource(
            __import__(
                "app.services.ctrader_client",
                fromlist=["_try_acquire_token_refresh_lock"],
            )._try_acquire_token_refresh_lock
        )
        self.assertIn("pg_try_advisory_xact_lock", lock_src)
        self.assertIn("_TOKEN_REFRESH_PG_LOCK_NS", lock_src)

    def test_feed_singleton_guard(self):
        from app.services import ctrader_price_feed as feed

        src = inspect.getsource(feed.launch_ctrader_feed)
        self.assertIn("_feed_starting", src)
        self.assertIn("_feed_launch_lock", src)
        self.assertIn("skip duplicate", src)

    def test_position_fetches_reread_token_on_auth_failure(self):
        import app.services.ctrader_client as cc

        open_src = inspect.getsource(cc._get_open_position_ids)
        deal_src = inspect.getsource(cc._fetch_deals_by_position_id)
        helper_src = inspect.getsource(cc._reread_token_for_auth_retry)

        self.assertIn("for auth_attempt in (1, 2)", open_src)
        self.assertIn("_reread_token_for_auth_retry", open_src)
        self.assertNotIn("_singleflight_forced_refresh", open_src)
        self.assertNotIn("refresh_user_ctrader_token", open_src)

        self.assertIn("for auth_attempt in (1, 2)", deal_src)
        self.assertIn("_reread_token_for_auth_retry", deal_src)
        self.assertNotIn("_singleflight_forced_refresh", deal_src)

        self.assertIn("_latest_ctrader_access_token", helper_src)
        self.assertNotIn("refresh_user_ctrader_token", helper_src)
        self.assertNotIn("_singleflight_forced_refresh", helper_src)

    def test_feed_reads_persisted_token_only(self):
        from app.services import ctrader_price_feed as feed

        src = inspect.getsource(feed._load_persisted_access_token)
        self.assertIn("_latest_ctrader_access_token", src)
        self.assertNotIn("refresh_user_ctrader_token", src)
        self.assertNotIn("request_ctrader_token_refresh", src)

        auth_src = inspect.getsource(feed._authenticate_stream)
        self.assertNotIn("force=True", auth_src)
        self.assertNotIn("request_ctrader_token_refresh", auth_src)

    def test_klines_pull_does_not_refresh(self):
        from app.services import ctrader_price_feed as feed

        src = inspect.getsource(feed.get_klines)
        self.assertNotIn("refresh_user_ctrader_token", src)

    def test_token_scheduler_owner_lock_registered(self):
        from app.advisory_lock_ids import (
            ALL_ADVISORY_LOCK_IDS,
            CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID,
        )

        self.assertIn(CTRADER_TOKEN_REFRESH_OWNER_LOCK_ID, ALL_ADVISORY_LOCK_IDS)

    def test_scheduled_refresh_early_remaining_default(self):
        from app.services.ctrader_client import _SCHEDULED_REFRESH_WHEN_REMAINING_S

        self.assertGreaterEqual(_SCHEDULED_REFRESH_WHEN_REMAINING_S, 86400)

    def test_no_reactive_forced_refresh_helpers(self):
        import app.services.ctrader_client as cc

        self.assertFalse(hasattr(cc, "_singleflight_forced_refresh"))
        self.assertFalse(hasattr(cc, "_refresh_auth_token_for_retry"))
        req_src = inspect.getsource(cc.request_ctrader_token_refresh)
        self.assertNotIn("force", req_src)

    def test_order_auth_failure_rereads_db_and_requests_refresh(self):
        import app.services.ctrader_client as cc

        src = inspect.getsource(cc.place_market_order_resilient)
        self.assertIn("ensure_ctrader_access_token_for_order", src)
        self.assertIn("_retry_order_after_account_auth_failure", src)
        self.assertIn("request_ctrader_token_refresh", inspect.getsource(cc._retry_order_after_account_auth_failure))
        self.assertNotIn("_singleflight_forced_refresh", src)
        self.assertNotIn("_notify_ctrader_relink_needed", src)


class TestNearExpiryAutoRecovery(unittest.IsolatedAsyncioTestCase):
    async def test_ensure_token_requests_refresh_when_near_expiry(self):
        from app.services import ctrader_client as cc

        prefs = MagicMock()
        prefs.ctrader_access_token = "old-access"
        prefs.ctrader_access_token_expires_at = datetime.utcnow() + timedelta(minutes=5)

        with patch.object(
            cc, "request_ctrader_token_refresh", new_callable=AsyncMock, return_value="new-access"
        ) as mock_refresh:
            token = await cc.ensure_ctrader_access_token_for_order(42, "old-access", prefs=prefs)

        self.assertEqual(token, "new-access")
        mock_refresh.assert_awaited_once()

    async def test_near_expiry_refresh_persists_rotated_tokens(self):
        from app.services import ctrader_client as cc

        prefs = MagicMock()
        prefs.user_id = 42
        prefs.ctrader_refresh_token = "old-refresh"
        prefs.ctrader_access_token = "old-access"
        prefs.ctrader_access_token_expires_at = datetime.utcnow() + timedelta(hours=12)

        oauth_response = {
            "accessToken": "new-access",
            "refreshToken": "new-refresh",
            "expiresIn": 86400 * 25,
        }

        db = MagicMock()
        db.execute.return_value.scalar.return_value = True

        with patch("app.database.SessionLocal", return_value=db), patch.object(
            cc, "_read_fresh_ctrader_prefs", return_value=prefs
        ), patch.object(
            cc, "refresh_access_token", new_callable=AsyncMock, return_value=oauth_response
        ), patch.object(
            cc, "_try_acquire_token_refresh_lock", new_callable=AsyncMock, return_value=True
        ), patch(
            "app.services.ctrader_price_feed.invalidate_stream_creds"
        ):
            token = await cc.refresh_user_ctrader_token(42)

        self.assertEqual(token, "new-access")
        self.assertEqual(prefs.ctrader_access_token, "new-access")
        self.assertEqual(prefs.ctrader_refresh_token, "new-refresh")
        db.commit.assert_called()


if __name__ == "__main__":
    unittest.main()
