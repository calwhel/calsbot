"""Source-aware metal kline/live drift limits."""
import unittest
import unittest.mock
from unittest.mock import AsyncMock, patch

from app.services import tradfi_prices as tp
from app.services.strategy_executor import _check_time_filter


class TestMetalKlineDrift(unittest.TestCase):
    def test_paxg_source_uses_widest_cap(self):
        self.assertEqual(tp.metal_kline_drift_limit("kraken"), tp.METAL_PAXG_KLINE_MAX_DRIFT_PCT)
        self.assertEqual(tp.metal_kline_drift_limit("coinbase"), tp.METAL_PAXG_KLINE_MAX_DRIFT_PCT)

    def test_ctrader_source_uses_spot_cap(self):
        self.assertEqual(tp.metal_kline_drift_limit("ctrader"), tp.METAL_SPOT_KLINE_MAX_DRIFT_PCT)

    def test_unknown_source_uses_strict_cap(self):
        self.assertEqual(tp.metal_kline_drift_limit(None), tp.METAL_KLINE_LIVE_MAX_DRIFT_PCT)
        self.assertEqual(tp.metal_kline_drift_limit("yahoo"), tp.METAL_KLINE_LIVE_MAX_DRIFT_PCT)

    def test_kraken_paxg_drift_within_paxg_cap(self):
        """Closed-bar vs PAXG live ~0.88% must pass the PAXG proxy threshold."""
        self.assertGreaterEqual(tp.METAL_PAXG_KLINE_MAX_DRIFT_PCT, 0.88)

    def test_source_cached_after_fetch(self):
        tp._METAL_KLINE_SOURCE_CACHE.clear()
        tp._METAL_KLINE_SOURCE_CACHE[("XAUUSD", "15m", 80)] = ("kraken", __import__("datetime").datetime.utcnow())
        self.assertEqual(tp.get_metal_kline_source("XAUUSD", "15m", 80), "kraken")


class TestKillzoneSessionFilter(unittest.TestCase):
    def test_london_kz_inside_window(self):
        with patch("app.services.strategy_executor.datetime") as mock_dt:
            mock_dt.utcnow.return_value = __import__("datetime").datetime(2026, 6, 8, 8, 0)
            ok = _check_time_filter({"session": {"sessions": ["london_kz"]}})
        self.assertTrue(ok)

    def test_london_kz_outside_window(self):
        with patch("app.services.strategy_executor.datetime") as mock_dt:
            mock_dt.utcnow.return_value = __import__("datetime").datetime(2026, 6, 8, 11, 0)
            ok = _check_time_filter({"session": {"sessions": ["london_kz"]}})
        self.assertFalse(ok)

    def test_any_kz_matches_ny_window(self):
        with patch("app.services.strategy_executor.datetime") as mock_dt:
            mock_dt.utcnow.return_value = __import__("datetime").datetime(2026, 6, 8, 13, 0)
            ok = _check_time_filter({"session": {"sessions": ["any_kz"]}})
        self.assertTrue(ok)


class TestMetalLiveForSource(unittest.IsolatedAsyncioTestCase):
    async def test_coinbase_live_uses_paxg_product(self):
        mock_resp = unittest.mock.MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"data": {"amount": "2650.5"}}
        mock_get = AsyncMock(return_value=mock_resp)
        mock_inst = unittest.mock.MagicMock()
        mock_inst.get = mock_get
        mock_ctx = unittest.mock.MagicMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_inst)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=mock_ctx):
            hit = await tp.get_metal_live_for_source("XAUUSD", "coinbase")
        self.assertEqual(hit, (2650.5, "coinbase"))
        call_url = mock_get.call_args[0][0]
        self.assertIn("PAXG-USD", call_url)


class TestMetalLiveFetchSource(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_stores_kline_source(self):
        tp._METAL_KLINE_SOURCE_CACHE.clear()
        with patch.object(
            tp, "_fetch_coinbase_metals_klines",
            new_callable=AsyncMock, return_value=[],
        ), patch.object(
            tp, "_fetch_kraken_metals_klines",
            new_callable=AsyncMock,
            return_value=[
                [1_700_000_000_000, 4320, 4330, 4310, 4328.0, 1.0]
            ] * 80,
        ), patch.object(
            tp, "_fetch_fmp_metals_klines", new_callable=AsyncMock, return_value=[],
        ), patch(
            "app.services.ctrader_price_feed.is_live", return_value=False,
        ):
            rows = await tp.fetch_metal_live_candles("XAUUSD", "15m", 80)
        self.assertEqual(len(rows), 80)
        self.assertEqual(tp.get_metal_kline_source("XAUUSD", "15m", 80), "kraken")


if __name__ == "__main__":
    unittest.main()
