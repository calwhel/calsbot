"""Telegram token order for forex trade alerts."""
import unittest
from unittest.mock import patch

from app.services.telegram_dm import bot_tokens_for_asset


class TelegramTokenOrderTest(unittest.TestCase):
    @patch("app.services.telegram_dm.bot_tokens", return_value=["main-tok", "forex-tok"])
    def test_forex_prefers_forex_token_first(self, _mock):
        order = bot_tokens_for_asset("forex")
        self.assertEqual(order, ["forex-tok", "main-tok"])

    @patch("app.services.telegram_dm.bot_tokens", return_value=["main-tok", "forex-tok"])
    def test_crypto_keeps_main_first(self, _mock):
        order = bot_tokens_for_asset("crypto")
        self.assertEqual(order, ["main-tok", "forex-tok"])


if __name__ == "__main__":
    unittest.main()
