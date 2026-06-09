"""Twelve Data feed — UTC window + rate limits."""
import unittest
from datetime import datetime
from unittest.mock import AsyncMock, patch

from app.services import twelve_data_feed as td


class TestTwelveDataWindow(unittest.TestCase):
    def test_active_window_midday_utc(self):
        noon = datetime(2026, 6, 8, 12, 0, 0)
        self.assertTrue(td.is_active_window(noon))

    def test_inactive_before_9_utc(self):
        early = datetime(2026, 6, 8, 8, 59, 0)
        self.assertFalse(td.is_active_window(early))

    def test_inactive_at_6pm_utc(self):
        close = datetime(2026, 6, 8, 18, 0, 0)
        self.assertFalse(td.is_active_window(close))

    def test_symbol_map_forex(self):
        self.assertEqual(td.to_twelve_data_symbol("EURUSD", "forex"), "EUR/USD")

    def test_symbol_map_gold(self):
        self.assertEqual(td.to_twelve_data_symbol("XAUUSD", "forex"), "XAU/USD")

    def test_symbol_map_nas100(self):
        self.assertEqual(td.to_twelve_data_symbol("NAS100", "index"), "NDX")


class TestTwelveDataAsync(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_quote_skips_outside_window(self):
        td._PRICE_CACHE.clear()
        with patch.object(td, "is_enabled", return_value=True), patch.object(
            td, "is_active_window", return_value=False
        ), patch("httpx.AsyncClient") as mock_client:
            px = await td.fetch_quote("EURUSD", "forex")
        self.assertIsNone(px)
        mock_client.assert_not_called()

    async def test_fetch_quote_returns_price(self):
        class _FakeResp:
            status_code = 200

            def json(self):
                return {"price": "1.0850"}

        class _FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                return False

            async def get(self, url, params=None):
                return _FakeResp()

        with patch.object(td, "is_enabled", return_value=True), patch.object(
            td, "is_active_window", return_value=True
        ), patch.object(td, "can_request", return_value=True), patch.object(
            td, "_api_key", return_value="test-key"
        ), patch("httpx.AsyncClient", _FakeClient):
            px = await td.fetch_quote("EURUSD", "forex")
        self.assertEqual(px, 1.0850)


if __name__ == "__main__":
    unittest.main()
