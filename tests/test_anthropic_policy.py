"""Tests for crypto vs forex Anthropic policy gates."""
import os
import unittest
from unittest.mock import patch

from app.services.anthropic_policy import (
    crypto_anthropic_enabled,
    is_forex_or_metals_symbol,
    portal_chart_ai_read_allowed,
)


class TestAnthropicPolicy(unittest.TestCase):
    def tearDown(self):
        for key in (
            "ENABLE_CRYPTO_ANTHROPIC",
            "DISABLE_CRYPTO_ANTHROPIC",
            "ENABLE_AUTO_TRADER_AI",
            "DISABLE_AUTO_TRADER_AI",
        ):
            os.environ.pop(key, None)

    def test_crypto_disabled_by_default(self):
        self.assertFalse(crypto_anthropic_enabled())

    def test_crypto_opt_in(self):
        os.environ["ENABLE_CRYPTO_ANTHROPIC"] = "1"
        self.assertTrue(crypto_anthropic_enabled())

    def test_disable_crypto_overrides_enable(self):
        os.environ["ENABLE_CRYPTO_ANTHROPIC"] = "1"
        os.environ["DISABLE_CRYPTO_ANTHROPIC"] = "1"
        self.assertFalse(crypto_anthropic_enabled())

    def test_forex_symbols_not_crypto_gated(self):
        self.assertTrue(is_forex_or_metals_symbol("XAUUSD"))
        self.assertTrue(is_forex_or_metals_symbol("EUR/USD"))
        self.assertFalse(is_forex_or_metals_symbol("BTCUSDT"))

    def test_portal_read_allows_forex_when_crypto_off(self):
        self.assertTrue(portal_chart_ai_read_allowed("XAUUSD"))
        self.assertFalse(portal_chart_ai_read_allowed("BTCUSDT"))

    def test_portal_read_allows_crypto_when_enabled(self):
        os.environ["ENABLE_CRYPTO_ANTHROPIC"] = "1"
        self.assertTrue(portal_chart_ai_read_allowed("BTCUSDT"))


class TestAiSignalFilterGate(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_signal_skips_claude_when_crypto_disabled(self):
        from app.services.ai_signal_filter import analyze_signal_with_ai

        result = await analyze_signal_with_ai({"symbol": "BTCUSDT", "direction": "LONG"})
        self.assertFalse(result["approved"])
        self.assertIn("disabled", result["reasoning"].lower())


if __name__ == "__main__":
    unittest.main()
