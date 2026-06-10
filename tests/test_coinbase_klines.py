"""Coinbase metal klines — retry, bounded window, stale fallback."""
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from app.services import tradfi_prices as tp


def _coinbase_raw(n: int = 80, close: float = 2650.0) -> list:
    now = int(datetime.utcnow().timestamp())
    return [
        [now - i * 60, close - 1, close + 1, close, close, 1.0]
        for i in range(n)
    ]


class TestCoinbaseKlines(unittest.IsolatedAsyncioTestCase):
    async def test_stale_cache_on_connect_timeout(self):
        tp._KLINE_CACHE.clear()
        stale_rows = [
            [1_700_000_000_000, 2650, 2651, 2649, 2650, 0.0],
        ]
        tp._KLINE_CACHE[("coinbase:PAXG-USD", "1m", 80)] = (
            stale_rows, datetime.utcnow() - timedelta(seconds=120),
        )

        import httpx

        async def _boom(*_a, **_k):
            raise httpx.ConnectTimeout("connect timed out")

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=_boom)

        with patch("httpx.AsyncClient", return_value=mock_client):
            rows = await tp._fetch_coinbase_metals_klines("XAUUSD", "1m", 80)

        self.assertEqual(rows, stale_rows)

    async def test_bounded_window_params(self):
        tp._KLINE_CACHE.clear()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"[]"
        mock_resp.json.return_value = _coinbase_raw(80)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("httpx.AsyncClient", return_value=mock_client):
            rows = await tp._fetch_coinbase_metals_klines("XAUUSD", "1m", 80)

        self.assertGreaterEqual(len(rows), 15)
        params = mock_client.get.call_args.kwargs.get("params") or mock_client.get.call_args[1].get("params")
        self.assertIn("start", params)
        self.assertIn("end", params)


if __name__ == "__main__":
    unittest.main()
