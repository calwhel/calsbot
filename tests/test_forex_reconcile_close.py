"""Forex broker close-reconcile — poll retry + 2-miss fallback close."""
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

from app.services.strategy_executor import (
    _FX_RECONCILE_MISSING,
    _classify_reconcile_fallback_exit,
    _reconcile_forex_closes,
)


class TestReconcileFallback(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FX_RECONCILE_MISSING.clear()

    async def test_classify_fallback_sl(self):
        w = {
            "symbol": "EURUSD",
            "direction": "LONG",
            "entry": 1.1000,
            "tp_price": 1.1100,
            "sl_price": 1.0950,
        }
        with patch("app.services.ctrader_price_feed.get_price", return_value=None):
            with patch(
                "app.services.tradfi_prices.get_price",
                new_callable=AsyncMock,
                return_value=None,
            ):
                outcome, exit_px = await _classify_reconcile_fallback_exit(w)
        self.assertEqual(outcome, "LOSS")
        self.assertEqual(exit_px, 1.0950)

    async def test_two_miss_closes_when_broker_flat(self):
        work = [{
            "exec_id": 501,
            "user_id": 9,
            "user": MagicMock(id=9),
            "position_id": 777001,
            "symbol": "XAUUSD",
            "direction": "LONG",
            "entry": 2650.0,
            "tp_price": 2660.0,
            "sl_price": 2640.0,
            "be_moved": False,
            "ctrader_account_id": "12345",
        }]
        _FX_RECONCILE_MISSING[501] = 1

        async def _fake_close(ex_id, outcome, exit_price, source=""):
            return True

        with patch(
            "app.services.strategy_executor._build_forex_reconcile_worklist",
            return_value=work,
        ), patch(
            "asyncio.to_thread",
            side_effect=lambda fn: fn(),
        ), patch(
            "app.db_resilience.run_with_db_retry",
            side_effect=lambda fn, **kw: fn(),
        ), patch(
            "app.services.ctrader_client.get_open_position_ids_for_user_with_retry",
            new_callable=AsyncMock,
            return_value=set(),
        ), patch(
            "app.services.ctrader_client.get_position_close_detail_for_user",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.strategy_executor._classify_reconcile_fallback_exit",
            new_callable=AsyncMock,
            return_value=("LOSS", 2640.0),
        ), patch(
            "app.services.strategy_executor._close_live_forex_execution_with_db_retry",
            new_callable=AsyncMock,
            side_effect=_fake_close,
        ) as mock_close, patch(
            "app.services.trade_management.reconcile_broker_pnl_for_recent_closes",
            new_callable=AsyncMock,
            return_value={},
        ), patch(
            "app.services.strategy_heal.expire_untracked_forex_opens_when_broker_empty",
            new_callable=AsyncMock,
        ):
            await _reconcile_forex_closes()

        mock_close.assert_awaited_once_with(
            501, "LOSS", 2640.0, source="ctrader-reconcile-fallback",
        )

    async def test_poll_fail_probes_deal_not_assumes_open(self):
        work = [{
            "exec_id": 502,
            "user_id": 9,
            "user": MagicMock(id=9),
            "position_id": 777002,
            "symbol": "EURUSD",
            "direction": "LONG",
            "entry": 1.10,
            "tp_price": 1.11,
            "sl_price": 1.09,
            "be_moved": False,
            "ctrader_account_id": "12345",
        }]

        with patch(
            "app.services.strategy_executor._build_forex_reconcile_worklist",
            return_value=work,
        ), patch(
            "asyncio.to_thread",
            side_effect=lambda fn: fn(),
        ), patch(
            "app.db_resilience.run_with_db_retry",
            side_effect=lambda fn, **kw: fn(),
        ), patch(
            "app.services.ctrader_client.get_open_position_ids_for_user_with_retry",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.ctrader_client.get_position_close_detail_for_user",
            new_callable=AsyncMock,
            return_value={"exit_price": 1.09, "outcome": "LOSS"},
        ), patch(
            "app.services.strategy_executor._close_live_forex_execution_with_db_retry",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_close, patch(
            "app.services.trade_management.reconcile_broker_pnl_for_recent_closes",
            new_callable=AsyncMock,
            return_value={},
        ), patch(
            "app.services.strategy_heal.expire_untracked_forex_opens_when_broker_empty",
            new_callable=AsyncMock,
        ):
            await _reconcile_forex_closes()

        mock_close.assert_awaited_once()
        args = mock_close.await_args[0]
        self.assertEqual(args[0], 502)
        self.assertEqual(args[1], "LOSS")


if __name__ == "__main__":
    unittest.main()
