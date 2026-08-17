"""Bitunix X promo — evergreen helpers + Aug campaign hard-push compatibility."""
import os
import sys
import types
import unittest
from datetime import datetime
from unittest import mock


class TestBitunixTwitterHelpers(unittest.TestCase):
    def test_deposit_legacy_april_campaign_has_ended(self):
        from app.services import twitter_poster as tw

        # New Aug window is active; April dates are gone from config.
        self.assertGreaterEqual(tw.BITUNIX_CAMPAIGN_START.month, 8)
        self.assertEqual(tw.BITUNIX_CAMPAIGN_START.year, 2026)

    def test_signup_link_uses_vip_code(self):
        from app.services import twitter_poster as tw

        self.assertIn("bitunix.com", tw.BITUNIX_SIGNUP_LINK)
        self.assertIn("vipCode=", tw.BITUNIX_SIGNUP_LINK)

    def test_twitter_ready_with_db_accounts_without_env(self):
        from app.services import twitter_poster as tw

        fake_account = mock.Mock(
            consumer_key="ck",
            consumer_secret="cs",
            access_token="at",
            access_token_secret="ats",
        )
        with mock.patch.object(tw, "get_all_twitter_accounts", return_value=[fake_account]):
            with mock.patch.dict(os.environ, {}, clear=True):
                self.assertTrue(tw._has_postable_twitter_accounts())
                self.assertTrue(tw._twitter_ready_to_post())

    def test_ensure_accounts_cover_schedule_adds_missing_types(self):
        from app.services import twitter_poster as tw

        account = mock.Mock()
        account.name = "ccally"
        account.get_post_types.return_value = ["bydfi_campaign"]
        account.set_post_types = mock.Mock()

        db = mock.MagicMock()
        db.query.return_value.filter.return_value.all.return_value = [account]

        fake_database = types.ModuleType("app.database")
        fake_database.SessionLocal = mock.Mock(return_value=db)
        fake_models = types.ModuleType("app.models")
        fake_models.TwitterAccount = mock.Mock()

        with mock.patch.dict(
            sys.modules,
            {"app.database": fake_database, "app.models": fake_models},
        ):
            tw.ensure_accounts_cover_schedule()

        account.set_post_types.assert_called_once()
        added = account.set_post_types.call_args[0][0]
        self.assertIn("bitunix_campaign", added)
        self.assertIn("top_gainer_ta", added)
        db.commit.assert_called_once()


class TestPostBitunixSignup(unittest.IsolatedAsyncioTestCase):
    async def test_post_bitunix_signup_posts_promo_link(self):
        from app.services import twitter_poster as tw

        poster = mock.Mock()
        poster.post_tweet.return_value = {"success": True, "tweet_id": "1"}

        with mock.patch.object(
            tw,
            "get_live_tickers_for_campaign",
            new=mock.AsyncMock(
                return_value={
                    "ticker1": "$BTC",
                    "ticker2": "$ETH",
                    "ticker3": "$SOL",
                    "pct1": "+2.1",
                    "pct2": "+1.4",
                }
            ),
        ), mock.patch.object(
            tw, "_get_ticker_suffix", new=mock.AsyncMock(return_value="")
        ), mock.patch.object(
            tw,
            "_ai_review_tweet",
            new=mock.AsyncMock(side_effect=lambda text, *_a, **_k: text),
        ), mock.patch.object(
            tw, "active_bitunix_promo_link", return_value=tw.BITUNIX_CAMPAIGN_LINK
        ):
            result = await tw.post_bitunix_signup(poster)

        self.assertTrue(result["success"])
        posted = poster.post_tweet.call_args[0][0]
        self.assertIn("bitunix.com", posted.lower())
        self.assertIn(tw.BITUNIX_CAMPAIGN_LINK, posted)

    async def test_expired_campaign_falls_back_to_signup(self):
        from app.services import twitter_poster as tw

        account_poster = mock.Mock()
        main_poster = mock.Mock()

        with mock.patch.object(
            tw, "post_bitunix_campaign", new=mock.AsyncMock(return_value=None)
        ), mock.patch.object(
            tw,
            "post_bitunix_signup",
            new=mock.AsyncMock(return_value={"success": True, "tweet_id": "9"}),
        ) as signup:
            result = await tw.post_with_account(
                account_poster, main_poster, "bitunix_campaign"
            )

        signup.assert_awaited_once()
        self.assertEqual(result["tweet_id"], "9")


if __name__ == "__main__":
    unittest.main()
