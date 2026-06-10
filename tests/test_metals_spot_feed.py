"""Dedicated XAUUSD / XAGUSD spot feed."""
import unittest
from unittest.mock import AsyncMock, MagicMock, patch


class MetalsSpotFeedTest(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_coinbase_gold(self):
        from app.services import metals_spot_feed as msf

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": {"amount": "2650.12", "base": "XAU", "currency": "USD"},
        }
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            px = await msf._fetch_coinbase("XAU-USD")
        self.assertAlmostEqual(px, 2650.12)

    async def test_poll_skips_when_tick_already_fresh(self):
        from app.services import metals_spot_feed as msf

        with patch.object(msf, "_has_fresh_tick", return_value=True):
            ok = await msf._poll_symbol("XAUUSD")
        self.assertTrue(ok)

    async def test_poll_stores_coinbase_when_kraken_fails(self):
        from app.services import metals_spot_feed as msf

        fetchers = {
            "coinbase": AsyncMock(return_value=4318.5),
            "kraken": AsyncMock(return_value=None),
        }
        with patch.object(msf, "_has_fresh_tick", return_value=False), patch.dict(
            msf._FETCHERS, fetchers, clear=True
        ), patch.object(msf, "_store") as mock_store:
            ok = await msf._poll_symbol("XAUUSD")
        self.assertTrue(ok)
        mock_store.assert_called_once_with("XAUUSD", 4318.5, "coinbase")


if __name__ == "__main__":
    unittest.main()
