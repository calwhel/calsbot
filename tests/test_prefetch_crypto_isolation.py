"""Prefetch — crypto/forex isolation, 2s budget, crypto kline cache."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import unittest
from unittest.mock import MagicMock, patch

from app.services import bitunix_market_data as cmd
from app.services import prefetch_provider_limits as ppl
from app.services.prefetch_fast import prefetch_fast_context
from app.services.prefetch_fast import PROVIDER_TIMEOUT_S, SYMBOL_BUDGET_S
from app.services.strategy_account_assignments import TRADFI_BROKER_ASSET_CLASSES
from app.services.strategy_executor import (
    _EXECUTOR_TRADFI_CLASSES,
    _PRICE_TA_CACHE,
    _prefetch_fallback_price_ta,
    _prefetch_price_ta_for_cycle,
    _snap_asset_class,
)


class TestPrefetchIsolation(unittest.TestCase):
    def setUp(self):
        _PRICE_TA_CACHE.clear()
    def test_tradfi_classes_include_metals_not_crypto(self):
        self.assertIn("metals", _EXECUTOR_TRADFI_CLASSES)
        self.assertIn("forex", _EXECUTOR_TRADFI_CLASSES)
        self.assertNotIn("crypto", _EXECUTOR_TRADFI_CLASSES)
        self.assertEqual(
            frozenset(TRADFI_BROKER_ASSET_CLASSES) | {"stock"},
            _EXECUTOR_TRADFI_CLASSES,
        )

    def test_snap_asset_class_metals_routes_tradfi(self):
        snap = {
            "config": {"asset_class": "metals", "universe": {"symbols": ["XAUUSD"]}},
            "_obj": MagicMock(asset_class="metals"),
        }
        self.assertEqual(_snap_asset_class(snap), "metals")
        self.assertIn(_snap_asset_class(snap), _EXECUTOR_TRADFI_CLASSES)

    def test_prefetch_budget_two_seconds(self):
        self.assertLessEqual(SYMBOL_BUDGET_S, 2.0)
        self.assertLessEqual(PROVIDER_TIMEOUT_S, 2.0)

    def test_crypto_kline_cache_peek(self):
        rows = [[1, 1, 2, 0.5, 1.5, 0], [2, 1.5, 2, 1, 1.8, 0]]
        cmd._store_crypto_kline_cache("BTCUSDT", "15m", 50, rows, "mexc")
        got, src = cmd.peek_cached_crypto_klines("BTCUSDT", "15m", 50)
        self.assertEqual(len(got), 2)
        self.assertEqual(src, "mexc")

    def test_crypto_timeout_uses_kline_cache_fallback(self):
        rows = [[i * 1000, 1, 2, 0.5, 1.0 + i * 0.01, 0] for i in range(20)]
        cmd._store_crypto_kline_cache("BTCUSDT", "15m", 200, rows, "mexc")
        got, src = _prefetch_fallback_price_ta("BTCUSDT", "crypto", "15m")
        self.assertIsNotNone(got)
        self.assertEqual(src, "mexc")

    def test_forex_prefetch_skips_crypto_symbols(self):
        snapshots = [
            {
                "id": 1,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["EURUSD"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
            {
                "id": 2,
                "config": {
                    "asset_class": "crypto",
                    "universe": {"symbols": ["BTCUSDT"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
        ]
        fetched = []

        async def _fake_fetch(sym, http_client, ac, **kwargs):
            fetched.append((sym, ac))
            return {"price": 1.0, "kline_source": ac}

        async def _run():
            http = MagicMock()
            with patch(
                "app.services.strategy_executor._fetch_price_and_ta",
                side_effect=_fake_fetch,
            ):
                return await _prefetch_price_ta_for_cycle(
                    snapshots, http, set(_EXECUTOR_TRADFI_CLASSES), label="FX",
                )

        stats = asyncio.run(_run())
        self.assertEqual(stats["unique_keys"], 1)
        self.assertEqual(fetched, [("EURUSD", "forex")])

    def test_prefetch_log_includes_executor_label(self):
        snapshots = [
            {
                "id": 1,
                "config": {
                    "asset_class": "forex",
                    "universe": {"symbols": ["EURUSD"]},
                    "entry_conditions": {"conditions": [{"timeframe": "15m"}]},
                },
            },
        ]

        async def _fake_fetch(sym, http_client, ac, **kwargs):
            return {"price": 1.0, "kline_source": "ctrader"}

        async def _run():
            http = MagicMock()
            with patch(
                "app.services.strategy_executor._fetch_price_and_ta",
                side_effect=_fake_fetch,
            ):
                with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
                    await _prefetch_price_ta_for_cycle(
                        snapshots, http, {"forex"}, label="FX Executor S1/3",
                    )
            return "\n".join(cm.output)

        logs = asyncio.run(_run())
        self.assertIn("executor=FX Executor S1/3", logs)

    def test_crypto_slot_wait_budget_aware_skip(self):
        async def _run():
            http = MagicMock()
            ppl._sems.clear()
            ppl._PROVIDER_LIMITS["crypto"] = 1
            sem = ppl._provider_sem("crypto")
            await sem.acquire()
            try:
                with patch("app.services.prefetch_fast.SYMBOL_BUDGET_S", 0.05):
                    with patch(
                        "app.services.bitunix_market_data._fetch_ticker_chain",
                        return_value=(
                            {
                                "lastPrice": "100.0",
                                "priceChangePercent": "0.0",
                                "quoteVolume": "1000",
                                "highPrice": "101.0",
                                "lowPrice": "99.0",
                            },
                            "binance",
                        ),
                    ):
                        async with prefetch_fast_context():
                            with self.assertLogs(
                                "app.services.bitunix_market_data",
                                level="INFO",
                            ) as cm:
                                t0 = asyncio.get_running_loop().time()
                                out = await cmd.fetch_crypto_price_and_ta(
                                    http,
                                    "BTCUSDT",
                                    prefetch=True,
                                )
                                elapsed = asyncio.get_running_loop().time() - t0
                return out, elapsed, "\n".join(cm.output)
            finally:
                sem.release()

        out, elapsed, logs = asyncio.run(_run())
        self.assertIsNone(out)
        self.assertLess(elapsed, 0.3)
        self.assertTrue(
            (
                "skipped: no fetch slot available in time" in logs
                or "no 15m klines available within budget" in logs
                or "price/TA fetch budget exceeded" in logs
            ),
        )


if __name__ == "__main__":
    unittest.main()
