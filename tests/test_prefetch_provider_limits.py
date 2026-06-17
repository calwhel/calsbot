"""Per-provider prefetch limits and 429 tracking."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import unittest

import app.services.prefetch_provider_limits as ppl
from app.services.prefetch_fast import prefetch_fast_context
from app.services.prefetch_provider_limits import (
    clear_prefetch_429,
    consume_prefetch_429,
    is_rate_limit_http,
    note_prefetch_429,
    prefetch_http_get,
    prefetch_provider_bucket,
    prefetch_provider_slot,
)


class TestPrefetchProviderLimits(unittest.TestCase):
    def test_rate_limit_codes(self):
        self.assertTrue(is_rate_limit_http(429))
        self.assertTrue(is_rate_limit_http(418))
        self.assertFalse(is_rate_limit_http(200))

    def test_provider_bucket_classification(self):
        self.assertEqual(prefetch_provider_bucket("BTCUSDT", "crypto"), "crypto")
        self.assertEqual(prefetch_provider_bucket("XAUUSD", "metals"), "kraken")
        self.assertEqual(prefetch_provider_bucket("NAS100", "index"), "yahoo")
        self.assertEqual(prefetch_provider_bucket("EURUSD", "forex"), "ctrader")

    def test_429_note_roundtrip(self):
        clear_prefetch_429()
        self.assertIsNone(consume_prefetch_429())
        note_prefetch_429("kraken")
        self.assertEqual(consume_prefetch_429(), "kraken")

    def test_provider_slot_serializes_under_limit(self):
        async def _run():
            order = []
            async with prefetch_fast_context():
                async def _task(n):
                    async with prefetch_provider_slot("kraken"):
                        order.append(f"start-{n}")
                        await asyncio.sleep(0.05)
                        order.append(f"end-{n}")

                ppl._sems.clear()
                ppl._PROVIDER_LIMITS["kraken"] = 1
                await asyncio.gather(_task(1), _task(2))

            self.assertEqual(order[0], "start-1")
            self.assertEqual(order[1], "end-1")
            self.assertEqual(order[2], "start-2")
            self.assertEqual(order[3], "end-2")

        asyncio.run(_run())


    def test_prefetch_http_get_retries_429(self):
        class _Resp:
            def __init__(self, code):
                self.status_code = code

        class _Client:
            def __init__(self):
                self.calls = 0

            async def get(self, url, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return _Resp(429)
                return _Resp(200)

        async def _run():
            clear_prefetch_429()
            client = _Client()
            async with prefetch_fast_context():
                ppl._sems.clear()
                resp = await prefetch_http_get("yahoo", client, "https://example.com")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(client.calls, 2)
            self.assertEqual(consume_prefetch_429(), "yahoo")

        asyncio.run(_run())


if __name__ == "__main__":
    unittest.main()
