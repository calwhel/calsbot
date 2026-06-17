"""Crypto universe_type=all cap — top-N by volume."""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio

from app.services.strategy_executor import (
    EXECUTOR_MAX_SYMBOLS_PER_STRATEGY,
    _cap_crypto_symbols_by_volume,
    _get_eligible_symbols,
)


def _tickers(n: int):
    return [
        {
            "symbol": f"COIN{i}USDT",
            "quoteVolume": float(1_000_000 * (n - i)),
            "priceChangePercent": 1.0,
        }
        for i in range(n)
    ]


class TestCapCryptoSymbols(unittest.TestCase):
    def test_caps_to_top_volume(self):
        tickers = _tickers(50)
        symbols = [t["symbol"] for t in tickers]
        capped = _cap_crypto_symbols_by_volume(
            symbols, tickers, {"type": "all"}, strategy_id=7,
        )
        self.assertEqual(len(capped), EXECUTOR_MAX_SYMBOLS_PER_STRATEGY)
        self.assertEqual(capped[0], "COIN0USDT")
        self.assertEqual(capped[-1], f"COIN{EXECUTOR_MAX_SYMBOLS_PER_STRATEGY - 1}USDT")

    def test_no_cap_when_under_limit(self):
        tickers = _tickers(5)
        symbols = [t["symbol"] for t in tickers]
        self.assertEqual(
            _cap_crypto_symbols_by_volume(symbols, tickers, {"type": "all"}),
            symbols,
        )


class TestGetEligibleSymbolsCap(unittest.IsolatedAsyncioTestCase):
    async def test_all_universe_capped(self):
        tickers = _tickers(80)
        bitunix = {t["symbol"] for t in tickers}

        async def _run():
            with patch(
                "app.services.strategy_executor._get_bitunix_symbols",
                new_callable=AsyncMock,
                return_value=bitunix,
            ):
                return await _get_eligible_symbols(
                    {"type": "all", "min_volume_usd": 0},
                    MagicMock(),
                    raw_tickers=tickers,
                    strategy_id=99,
                )

        with self.assertLogs("app.services.strategy_executor", level="INFO") as cm:
            syms = await _run()
        self.assertEqual(len(syms), EXECUTOR_MAX_SYMBOLS_PER_STRATEGY)
        self.assertTrue(
            any("strategy=99 universe all capped" in line for line in cm.output),
        )


if __name__ == "__main__":
    unittest.main()
