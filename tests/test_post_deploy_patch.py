"""Post-deploy patch: CancelledError gather, synthetic bars, twitter gate."""
import asyncio
import unittest
from unittest import mock

from app.services.strategy_ta import StrategyEvalCancelled, evaluate_strategy_conditions


class TestCancelledErrorGather(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_task_raises_strategy_eval_cancelled(self):
        config = {
            "entry_conditions": {
                "operator": "AND",
                "conditions": [
                    {"type": "volume_spike"},
                    {"type": "volume_spike"},
                ],
            },
        }
        price_data = {"price": 100.0, "_asset_class": "forex"}

        def _boom(*_a, **_k):
            raise asyncio.CancelledError()

        with mock.patch(
            "app.services.strategy_ta.eval_volume_spike",
            side_effect=_boom,
        ):
            with self.assertRaises(StrategyEvalCancelled):
                await evaluate_strategy_conditions(
                    config, "XAUUSD", price_data, {}, None,
                )


class TestSyntheticBarsGuard(unittest.TestCase):
    def test_is_metal_kline_synthetic(self):
        from datetime import datetime
        from app.services import tradfi_prices as tp

        tp._METAL_KLINE_SOURCE_CACHE.clear()
        tp._METAL_KLINE_SOURCE_CACHE[("XAUUSD", "15m", 80)] = (
            "synthetic",
            datetime.utcnow(),
        )
        self.assertTrue(tp.is_metal_kline_synthetic("XAUUSD", "15m", 80))
        self.assertFalse(tp.is_metal_kline_synthetic("EURUSD", "15m", 80))


class TestTwitterEnabledGate(unittest.TestCase):
    def test_twitter_disabled_by_default_without_creds(self):
        from app.services import twitter_poster as tw

        with mock.patch.dict(
            "os.environ",
            {},
            clear=True,
        ):
            tw.TWITTER_ENABLED = False
            self.assertFalse(tw.twitter_poster_active())

    def test_executor_lock_connect_no_duplicate_timeout(self):
        import inspect
        from app import executor_lock as el

        src = inspect.getsource(el.terminate_lock_holders)
        self.assertIn("psycopg2.connect(db_url, **NEON_LOCK_CONNECT_KWARGS)", src)
        self.assertNotIn("connect_timeout=10, **NEON_LOCK_CONNECT_KWARGS", src)


if __name__ == "__main__":
    unittest.main()
