"""X mover auto-replies — TA comments under high-engagement cashtag tweets."""
import unittest
from unittest import mock


class TestMoverReplyCopy(unittest.TestCase):
    def test_reply_text_has_no_links_or_affiliate(self):
        from app.services import twitter_poster as tw

        for _ in range(30):
            text = tw._build_mover_reply_text(
                "SOL",
                12.4,
                {"rsi": 68.2, "trend": "bullish", "vol_ratio": 2.1},
            )
            low = text.lower()
            self.assertNotIn("http", low)
            self.assertNotIn("bitunix", low)
            self.assertNotIn("sign up", low)
            self.assertIn("$SOL", text)
            self.assertLessEqual(len(text), 220)

    def test_post_tweet_passes_in_reply_to(self):
        from app.services.twitter_poster import MultiAccountPoster

        account = mock.Mock(
            id=1,
            name="ccally",
            consumer_key="enc",
            consumer_secret="enc",
            access_token="enc",
            access_token_secret="enc",
            bearer_token=None,
        )
        poster = MultiAccountPoster.__new__(MultiAccountPoster)
        poster.account = account
        poster.account_id = 1
        poster.name = "ccally"
        poster.client = mock.Mock()
        poster.api_v1 = None
        poster.client.create_tweet.return_value = mock.Mock(data={"id": "99"})

        with mock.patch(
            "app.services.twitter_poster._strip_extra_cashtags",
            side_effect=lambda t: t,
        ):
            result = poster.post_tweet("nice move", in_reply_to_tweet_id="12345")

        self.assertTrue(result["success"])
        kwargs = poster.client.create_tweet.call_args.kwargs
        self.assertEqual(kwargs.get("in_reply_to_tweet_id"), "12345")
        self.assertEqual(result["in_reply_to"], "12345")


class TestMoverReplyCycle(unittest.IsolatedAsyncioTestCase):
    async def test_cycle_respects_daily_cap(self):
        from app.services import twitter_poster as tw

        with mock.patch.object(tw, "TWITTER_ENABLED", True), \
                mock.patch.object(tw, "TWITTER_AUTO_REPLY_ENABLED", True), \
                mock.patch.object(tw, "_twitter_ready_to_post", return_value=True), \
                mock.patch.object(tw, "_ensure_mover_replies_table"), \
                mock.patch.object(tw, "_mover_replies_today_count", return_value=99), \
                mock.patch.object(tw, "MAX_MOVER_REPLIES_PER_DAY", 8):
            n = await tw.run_mover_reply_cycle()
        self.assertEqual(n, 0)

    async def test_cycle_posts_reply_without_links(self):
        from app.services import twitter_poster as tw

        account = mock.Mock(name="ccally")
        account.name = "ccally"
        poster = mock.Mock()
        poster.client = mock.Mock()
        poster.name = "ccally"
        poster.post_tweet.return_value = {
            "success": True,
            "tweet_id": "reply1",
        }

        targets = [{
            "tweet_id": "parent1",
            "symbol": "PEPE",
            "likes": 120,
            "text": "$PEPE ripping",
            "author_id": "999",
        }]

        with mock.patch.object(tw, "TWITTER_ENABLED", True), \
                mock.patch.object(tw, "TWITTER_AUTO_REPLY_ENABLED", True), \
                mock.patch.object(tw, "_twitter_ready_to_post", return_value=True), \
                mock.patch.object(tw, "_ensure_mover_replies_table"), \
                mock.patch.object(tw, "_mover_replies_today_count", return_value=0), \
                mock.patch.object(tw, "MOVER_REPLIES_PER_CYCLE", 1), \
                mock.patch.object(tw, "get_all_twitter_accounts", return_value=[account]), \
                mock.patch.object(tw, "get_account_poster", return_value=poster), \
                mock.patch.object(
                    tw,
                    "_fetch_mexc_tickers",
                    new=mock.AsyncMock(return_value=[{"symbol": "PEPE", "change": 18.0, "volume": 5e6}]),
                ), \
                mock.patch.object(tw, "_update_daily_gainers"), \
                mock.patch.object(tw, "_get_own_x_user_id", new=mock.AsyncMock(return_value="1")), \
                mock.patch.object(
                    tw, "_find_mover_reply_targets", new=mock.AsyncMock(return_value=targets)
                ), \
                mock.patch.object(
                    tw,
                    "_quick_mover_ta_bits",
                    new=mock.AsyncMock(return_value={"rsi": 70.0, "trend": "bullish", "vol_ratio": 2.0}),
                ), \
                mock.patch.object(tw, "_save_mover_reply") as save, \
                mock.patch.object(
                    tw, "notify_admin_post_result", new=mock.AsyncMock()
                ), \
                mock.patch.object(tw.asyncio, "sleep", new=mock.AsyncMock()), \
                mock.patch.object(tw.asyncio, "to_thread", new=mock.AsyncMock(
                    side_effect=lambda fn, *a, **k: fn(*a, **k)
                )):
            n = await tw.run_mover_reply_cycle()

        self.assertEqual(n, 1)
        save.assert_called_once()
        posted_text = poster.post_tweet.call_args[0][0]
        self.assertNotIn("http", posted_text.lower())
        self.assertEqual(poster.post_tweet.call_args[0][2], "parent1")


if __name__ == "__main__":
    unittest.main()
