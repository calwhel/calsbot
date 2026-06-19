"""cTrader OAuth single-flight refresh — deploy-survival guards."""
import inspect
from pathlib import Path
import unittest


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

    def test_position_fetches_refresh_and_retry_on_auth_failure(self):
        import app.services.ctrader_client as cc

        open_src = inspect.getsource(cc._get_open_position_ids)
        deal_src = inspect.getsource(cc._fetch_deals_by_position_id)
        helper_src = inspect.getsource(cc._refresh_auth_token_for_retry)
        singleflight_src = inspect.getsource(cc._singleflight_forced_refresh)

        self.assertIn("for auth_attempt in (1, 2)", open_src)
        self.assertIn("_refresh_auth_token_for_retry", open_src)
        self.assertIn("open-positions auth failed", open_src)
        self.assertIn("asyncio.timeout(", open_src)
        self.assertIn("_account_auth_in_cooldown", open_src)
        self.assertIn("_mark_account_auth_unhealthy", open_src)

        self.assertIn("for auth_attempt in (1, 2)", deal_src)
        self.assertIn("_refresh_auth_token_for_retry", deal_src)
        self.assertIn("deal-by-position auth failed", deal_src)
        self.assertIn("asyncio.timeout(", deal_src)
        self.assertIn("_mark_account_auth_unhealthy", deal_src)

        self.assertIn("_singleflight_forced_refresh", helper_src)
        self.assertIn("_mark_account_auth_unhealthy", helper_src)
        self.assertIn("_AUTH_REFRESH_COOLDOWN_S", singleflight_src)

    def test_feed_does_not_call_refresh_directly(self):
        from app.services import ctrader_price_feed as feed

        src = inspect.getsource(feed._maybe_refresh_access_token)
        self.assertNotIn("refresh_user_ctrader_token", src)
        self.assertIn("request_ctrader_token_refresh", src)

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


if __name__ == "__main__":
    unittest.main()
