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

    def test_campaign_link_and_image(self):
        from app.services import twitter_poster as tw
        import os

        self.assertIn("ENWeeklyCampaign0817", tw.BITUNIX_CAMPAIGN_LINK)
        self.assertIn("vipCode=fgq74890", tw.BITUNIX_CAMPAIGN_LINK)
        self.assertTrue(os.path.exists(tw.BITUNIX_CAMPAIGN_IMAGE), tw.BITUNIX_CAMPAIGN_IMAGE)

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


if __name__ == "__main__":
    unittest.main()
