"""Prefetch fast-path — hard timeouts, no backoff sleeps."""
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from app.services import fmp_price_feed as fmp
from app.services.prefetch_fast import (
    PROVIDER_TIMEOUT_S,
    SYMBOL_BUDGET_S,
    prefetch_fast_active,
    prefetch_fast_context,
    provider_timeout_s,
)


class TestPrefetchFast(unittest.TestCase):
    async def _async_provider_timeout_caps(self):
        self.assertEqual(provider_timeout_s(20.0), 20.0)
        async with prefetch_fast_context():
            self.assertEqual(provider_timeout_s(20.0), PROVIDER_TIMEOUT_S)

    def test_provider_timeout_caps_in_prefetch_mode(self):
        asyncio.run(self._async_provider_timeout_caps())

    def test_symbol_budget_default(self):
        self.assertGreater(SYMBOL_BUDGET_S, 0)
        self.assertLessEqual(SYMBOL_BUDGET_S, 3.0)

    async def _async_prefetch_flag(self):
        self.assertFalse(prefetch_fast_active())
        async with prefetch_fast_context():
            self.assertTrue(prefetch_fast_active())
        self.assertFalse(prefetch_fast_active())

    def test_prefetch_context_sets_flag(self):
        asyncio.run(self._async_prefetch_flag())

    async def _async_fmp_skip_on_backoff(self):
        fmp._FMP_BACKOFF_UNTIL = datetime.utcnow() + timedelta(seconds=300)
        ok = await fmp._fmp_rate_limit_wait()
        self.assertFalse(ok)
        fmp._FMP_BACKOFF_UNTIL = None

    def test_fmp_rate_limit_skips_during_backoff(self):
        asyncio.run(self._async_fmp_skip_on_backoff())


if __name__ == "__main__":
    unittest.main()
