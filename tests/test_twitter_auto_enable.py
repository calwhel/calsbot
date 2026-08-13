"""TWITTER_ENABLED auto-on when credentials exist."""
import os
import unittest
from unittest import mock


class TestTwitterAutoEnable(unittest.TestCase):
    def test_explicit_false_stays_off_even_with_creds(self):
        from app.services import twitter_poster as tw

        env = {
            "TWITTER_ENABLED": "false",
            "TWITTER_BEARER_TOKEN": "bearer",
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(tw.twitter_posting_enabled())

    def test_unset_enables_when_bearer_present(self):
        from app.services import twitter_poster as tw

        with mock.patch.dict(
            os.environ, {"TWITTER_BEARER_TOKEN": "bearer"}, clear=True
        ):
            self.assertTrue(tw.twitter_posting_enabled())

    def test_unset_enables_when_db_accounts_exist(self):
        from app.services import twitter_poster as tw

        fake = mock.Mock(
            consumer_key="ck",
            consumer_secret="cs",
            access_token="at",
            access_token_secret="ats",
        )
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(tw, "get_all_twitter_accounts", return_value=[fake]):
            self.assertTrue(tw.twitter_posting_enabled())
            self.assertTrue(tw.twitter_poster_active())

    def test_unset_stays_off_without_creds(self):
        from app.services import twitter_poster as tw

        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(tw, "get_all_twitter_accounts", return_value=[]):
            self.assertFalse(tw.twitter_posting_enabled())
            self.assertFalse(tw.twitter_poster_active())


if __name__ == "__main__":
    unittest.main()
