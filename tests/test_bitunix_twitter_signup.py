"""Influencer-first X auto-post + soft Bitunix signup."""
import os
import sys
import types
import unittest
from datetime import datetime
from unittest import mock


class TestInfluencerSchedule(unittest.TestCase):
    def test_schedule_is_influencer_first_with_soft_bitunix(self):
        from app.services import twitter_poster as tw

        types = [pt for _, _, pt in tw.POST_SCHEDULE]
        soft = types.count("bitunix_signup")
        value = len(types) - soft

        self.assertEqual(soft, 2, "exactly 2 soft Bitunix slots/day")
        self.assertGreaterEqual(value, 7, "majority of slots are value/influencer content")
        self.assertNotIn("bydfi_campaign", types)
        self.assertNotIn("yubit_campaign", types)

        # Core view-driving formats present
        for needed in (
            "top_gainer_ta",
            "early_gainer",
            "memecoin",
            "market_take",
            "quick_ta",
            "high_viewing",
            "altcoin_movers",
        ):
            self.assertIn(needed, types)

        # Soft CTAs are not clustered at the start of the day
        first_soft_idx = types.index("bitunix_signup")
        self.assertGreaterEqual(first_soft_idx, 3)

    def test_deposit_campaign_window_has_ended(self):
        from app.services import twitter_poster as tw

        now = datetime(2026, 8, 13)
        self.assertTrue(now > tw.BITUNIX_CAMPAIGN_END)
        self.assertTrue(now > tw.BYDFI_CAMPAIGN_END)
        self.assertTrue(now > tw.YUBIT_CAMPAIGN_END)

    def test_signup_templates_are_soft_not_hard_promo(self):
        from app.services import twitter_poster as tw

        blob = " ".join(t["text"].lower() for t in tw.BITUNIX_SIGNUP_TEMPLATES)
        self.assertIn("bitunix", blob)
        self.assertIn("{link}", blob)
        for banned in (
            "sign up now",
            "deposit bonus",
            "slots",
            "first come",
            "voucher",
            "vip code applied",
            "register here",
        ):
            self.assertNotIn(banned, blob)

    def test_signup_link_uses_referral_env(self):
        from app.services import twitter_poster as tw

        self.assertIn("bitunix.com/register", tw.BITUNIX_SIGNUP_LINK)
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
        self.assertIn("bitunix_signup", added)
        self.assertIn("top_gainer_ta", added)
        self.assertIn("high_viewing", added)
        self.assertIn("market_take", added)
        self.assertIn("bydfi_campaign", added)  # preserves existing
        db.commit.assert_called_once()


class TestPostBitunixSignup(unittest.IsolatedAsyncioTestCase):
    async def test_post_bitunix_signup_posts_referral_link(self):
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
        ):
            result = await tw.post_bitunix_signup(poster)

        self.assertTrue(result["success"])
        posted = poster.post_tweet.call_args[0][0]
        self.assertIn("bitunix.com", posted.lower())
        self.assertIn(tw.BITUNIX_SIGNUP_LINK, posted)

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

    async def test_tradehub_promo_no_longer_hijacks_to_bitunix(self):
        from app.services import twitter_poster as tw

        account_poster = mock.Mock()
        main_poster = mock.Mock()

        with mock.patch.object(
            tw,
            "post_tradehub_promo",
            new=mock.AsyncMock(return_value={"success": True, "tweet_id": "edu"}),
        ) as edu, mock.patch.object(
            tw, "post_bitunix_signup", new=mock.AsyncMock()
        ) as signup:
            result = await tw.post_with_account(
                account_poster, main_poster, "tradehub_promo"
            )

        edu.assert_awaited_once()
        signup.assert_not_awaited()
        self.assertEqual(result["tweet_id"], "edu")


if __name__ == "__main__":
    unittest.main()
