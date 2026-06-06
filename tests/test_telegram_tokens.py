import os
import unittest
from unittest.mock import patch

from app.services import telegram_tokens


class TelegramTokensTest(unittest.TestCase):
    def test_should_not_run_forex_when_tokens_match(self):
        with patch.object(telegram_tokens, "main_bot_token", return_value="tok-a"):
            with patch.object(telegram_tokens, "forex_bot_token", return_value="tok-a"):
                self.assertFalse(telegram_tokens.should_run_forex_poller())

    def test_should_run_forex_when_distinct(self):
        with patch.object(telegram_tokens, "main_bot_token", return_value="tok-a"):
            with patch.object(telegram_tokens, "forex_bot_token", return_value="tok-b"):
                self.assertTrue(telegram_tokens.should_run_forex_poller())

    def test_main_bot_token_prefers_settings(self):
        with patch("app.config.settings") as mock_settings:
            mock_settings.TELEGRAM_BOT_TOKEN = "from-settings"
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "from-env"}, clear=False):
                self.assertEqual(telegram_tokens.main_bot_token(), "from-settings")


if __name__ == "__main__":
    unittest.main()
