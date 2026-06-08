"""Bitunix public market-data fallback mapping (MEXC-shape reshaping).

Crypto data (tickers + klines) falls back to Bitunix when MEXC/Binance are
geo-blocked on Railway. These tests pin the reshaping so consumers that expect
the MEXC payload format keep working — without hitting the network.
"""
import asyncio
import unittest


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload


class _FakeClient:
    """Minimal async stand-in for httpx.AsyncClient returning canned payloads."""

    def __init__(self, tickers_payload, kline_payload):
        self._tickers = tickers_payload
        self._kline = kline_payload
        self.calls = []

    async def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if url.endswith("/tickers"):
            return _FakeResp(self._tickers)
        if url.endswith("/kline"):
            return _FakeResp(self._kline)
        return _FakeResp({}, status=404)


_TICKERS = {
    "code": 0,
    "msg": "Success",
    "data": [
        {"symbol": "BTCUSDT", "lastPrice": "63350.7", "open": "62276.3",
         "last": "63350.0", "quoteVol": "2519991545.9", "baseVol": "39000",
         "high": "64228.6", "low": "61152.3"},
        {"symbol": "ETHUSDT", "lastPrice": "1679.51", "open": "1634.01",
         "quoteVol": "2120489643.4", "high": "1721.13", "low": "1602.44"},
    ],
}

# Bitunix returns NEWEST-first dicts.
_KLINE = {
    "code": 0,
    "msg": "Success",
    "data": [
        {"open": "63496.9", "high": "63558.1", "low": "63271.3", "close": "63322.9",
         "quoteVol": "601.1", "baseVol": "38118555.4", "time": "1780938000000"},
        {"open": "63322.9", "high": "63620", "low": "63323", "close": "63598.7",
         "quoteVol": "298.2", "baseVol": "18931765.7", "time": "1780938900000"},
    ],
}


class TestBitunixMarketData(unittest.TestCase):
    def setUp(self):
        from app.services import bitunix_market_data as bm
        self.bm = bm
        self.client = _FakeClient(_TICKERS, _KLINE)

    def _run(self, coro):
        return asyncio.run(coro)

    def test_tickers_mexc_shape(self):
        tks = self._run(self.bm.fetch_tickers(self.client))
        self.assertEqual(len(tks), 2)
        btc = next(t for t in tks if t["symbol"] == "BTCUSDT")
        # MEXC consumers read these exact keys.
        for key in ("lastPrice", "priceChangePercent", "quoteVolume", "highPrice", "lowPrice"):
            self.assertIn(key, btc)
        self.assertAlmostEqual(float(btc["lastPrice"]), 63350.7)
        # priceChangePercent derived from open vs last: (63350.7-62276.3)/62276.3*100
        self.assertAlmostEqual(float(btc["priceChangePercent"]), (63350.7 - 62276.3) / 62276.3 * 100, places=4)
        self.assertAlmostEqual(float(btc["quoteVolume"]), 2519991545.9)

    def test_klines_mexc_shape_ascending(self):
        kl = self._run(self.bm.fetch_klines(self.client, "BTCUSDT", "15m", 10))
        self.assertEqual(len(kl), 2)
        # Reordered oldest-first.
        self.assertTrue(kl[0][0] <= kl[1][0])
        self.assertEqual(kl[0][0], 1780938000000)
        # row[1..4] = OHLC, float()-able like MEXC strings.
        self.assertAlmostEqual(float(kl[0][4]), 63322.9)   # close of oldest
        self.assertAlmostEqual(float(kl[1][4]), 63598.7)   # close of newest
        self.assertEqual(len(kl[0]), 8)

    def test_kline_interval_alias(self):
        self._run(self.bm.fetch_klines(self.client, "BTCUSDT", "60m", 5))
        # 60m must be mapped to a Bitunix-supported token (1h).
        kline_call = next(c for c in self.client.calls if c[0].endswith("/kline"))
        self.assertEqual(kline_call[1]["interval"], "1h")

    def test_empty_payload_safe(self):
        client = _FakeClient({"code": 0, "data": []}, {"code": 0, "data": []})
        self.assertEqual(self._run(self.bm.fetch_tickers(client)), [])
        self.assertEqual(self._run(self.bm.fetch_klines(client, "BTCUSDT", "15m", 5)), [])


if __name__ == "__main__":
    unittest.main()
