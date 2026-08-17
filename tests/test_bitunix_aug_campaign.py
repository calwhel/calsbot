"""Aug 2026 Bitunix x TradeHub campaign hard-push."""
import unittest
from datetime import datetime
from unittest import mock


class TestAugBitunixCampaign(unittest.TestCase):
    def test_campaign_window_covers_mid_august(self):
        from app.services import twitter_poster as tw

        self.assertTrue(tw.bitunix_campaign_is_active(datetime(2026, 8, 17, 12, 0, 0)))
        self.assertTrue(tw.bitunix_campaign_is_active(datetime(2026, 8, 25, 0, 0, 0)))
        self.assertFalse(tw.bitunix_campaign_is_active(datetime(2026, 9, 1, 0, 0, 0)))
        self.assertFalse(tw.bitunix_campaign_is_active(datetime(2026, 8, 10, 0, 0, 0)))

    def test_campaign_link_hardcoded_no_env(self):
        from app.services import twitter_poster as tw
        import os

        expected = "https://www.bitunix.com/activity/basic/ENWeeklyCampaign0817?vipCode=fgq74890"
        self.assertEqual(tw.BITUNIX_CAMPAIGN_LINK, expected)
        # Env overrides must not change the hardcoded campaign link
        with mock.patch.dict(
            os.environ,
            {
                "BITUNIX_CAMPAIGN_URL": "https://evil.example/wrong",
                "BITUNIX_REFERRAL_URL": "https://evil.example/wrong2",
            },
            clear=False,
        ):
            import importlib
            importlib.reload(tw)
            self.assertEqual(tw.BITUNIX_CAMPAIGN_LINK, expected)
            self.assertEqual(
                tw.BITUNIX_SIGNUP_LINK,
                "https://www.bitunix.com/register?vipCode=fgq74890",
            )

    def test_campaign_image_works_without_env(self):
        from app.services import twitter_poster as tw
        import os

        with mock.patch.dict(os.environ, {"BITUNIX_CAMPAIGN_IMAGE": "/nonexistent/fake.png"}, clear=False):
            img = tw.resolve_bitunix_campaign_image()
        self.assertTrue(os.path.exists(img), img)
        self.assertIn("bitunix_campaign", os.path.basename(img).lower())
        self.assertTrue(img.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))

    def test_schedule_is_campaign_heavy(self):
        from app.services import twitter_poster as tw

        types = [pt for _, _, pt in tw.POST_SCHEDULE]
        self.assertGreaterEqual(types.count("bitunix_campaign"), 6)
        self.assertGreaterEqual(len([t for t in types if t != "bitunix_campaign"]), 3)
        self.assertNotIn("bydfi_campaign", types)

    def test_templates_mention_aug_rewards(self):
        from app.services import twitter_poster as tw

        blob = " ".join(t["text"].lower() for t in tw.CAMPAIGN_TEMPLATES)
        self.assertTrue("20,000" in blob or "20000" in blob)
        self.assertIn("aug", blob)
        self.assertIn("{link}", blob)
        self.assertNotIn("april", blob)

    def test_active_promo_link_switches(self):
        from app.services import twitter_poster as tw

        with mock.patch.object(tw, "bitunix_campaign_is_active", return_value=True):
            self.assertEqual(tw.active_bitunix_promo_link(), tw.BITUNIX_CAMPAIGN_LINK)
        with mock.patch.object(tw, "bitunix_campaign_is_active", return_value=False):
            self.assertEqual(tw.active_bitunix_promo_link(), tw.BITUNIX_SIGNUP_LINK)

    def test_influencer_personalities_weighted(self):
        from app.services import twitter_poster as tw

        names = {tw._pick_personality()["name"] for _ in range(80)}
        self.assertTrue(
            {"crypto_influencer", "alpha_poster"} & names,
            f"expected influencer voices in sample, got {names}",
        )


class TestCampaignRouting(unittest.IsolatedAsyncioTestCase):
    async def test_signup_slot_routes_to_campaign_while_active(self):
        from app.services import twitter_poster as tw

        with mock.patch.object(tw, "bitunix_campaign_is_active", return_value=True), \
                mock.patch.object(
                    tw,
                    "post_bitunix_campaign",
                    new=mock.AsyncMock(return_value={"success": True, "tweet_id": "c1"}),
                ) as camp, mock.patch.object(
                    tw, "post_bitunix_signup", new=mock.AsyncMock()
                ) as signup:
            result = await tw.post_with_account(mock.Mock(), mock.Mock(), "bitunix_signup")

        camp.assert_awaited_once()
        signup.assert_not_awaited()
        self.assertEqual(result["tweet_id"], "c1")

    async def test_campaign_prefers_ai_copy(self):
        from app.services import twitter_poster as tw

        poster = mock.Mock()
        poster.upload_media = mock.Mock(return_value="media1")
        poster.post_tweet = mock.Mock(return_value={"success": True, "tweet_id": "t9"})

        with mock.patch.object(tw, "BITUNIX_CAMPAIGN_START", datetime(2020, 1, 1)), \
                mock.patch.object(tw, "BITUNIX_CAMPAIGN_END", datetime(2099, 1, 1)), \
                mock.patch.object(
                    tw, "get_trending_hashtags", new=mock.AsyncMock(return_value="")
                ), \
                mock.patch.object(
                    tw,
                    "get_live_tickers_for_campaign",
                    new=mock.AsyncMock(
                        return_value={
                            "ticker1": "$PEPE",
                            "ticker2": "$WIF",
                            "ticker3": "$BONK",
                            "pct1": "12",
                            "pct2": "8",
                            "pct3": "5",
                        }
                    ),
                ), \
                mock.patch.object(
                    tw,
                    "generate_bitunix_campaign_tweet",
                    new=mock.AsyncMock(
                        return_value=(
                            "deposit slots on bitunix are thin — claim before Aug 31\n\n"
                            + tw.BITUNIX_CAMPAIGN_LINK
                        )
                    ),
                ), \
                mock.patch.object(
                    tw, "_get_ticker_suffix", new=mock.AsyncMock(return_value="")
                ), \
                mock.patch.object(
                    tw, "_ai_review_tweet", new=mock.AsyncMock(side_effect=lambda t, *a, **k: t)
                ), \
                mock.patch.object(tw, "resolve_bitunix_campaign_image", return_value="/tmp/nope.png"):
            result = await tw.post_bitunix_campaign(poster)

        self.assertTrue(result["success"])
        posted = poster.post_tweet.call_args[0][0]
        self.assertIn(tw.BITUNIX_CAMPAIGN_LINK, posted)
        self.assertIn("deposit slots", posted.lower())


if __name__ == "__main__":
    unittest.main()
