"""Forex broker close-reconcile — deal fetch + estimated exit timeout."""
import os
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

from app.services.strategy_executor import (
    _FX_RECONCILE_ESTIMATE_AFTER_S,
    _FX_RECONCILE_MISSING,
    _FX_RECONCILE_MISSING_SINCE,
    _clear_reconcile_missing,
    _estimate_reconcile_exit,
    _reconcile_forex_closes,
)


class TestReconcileFallback(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        _FX_RECONCILE_MISSING.clear()
        _FX_RECONCILE_MISSING_SINCE.clear()

    async def test_estimate_exit_uses_market_price(self):
        w = {
            "symbol": "EURUSD",
            "direction": "LONG",
            "entry": 1.1000,
        }
        with patch("app.services.ctrader_price_feed.get_price", return_value=1.1050):
            outcome, px = await _estimate_reconcile_exit(w)
        self.assertEqual(outcome, "WIN")
        self.assertEqual(px, 1.1050)

    def _patch_reconcile(self, work, **extra):
        patches = {
            "worklist": patch(
                "app.services.strategy_executor._build_forex_reconcile_worklist",
                return_value=work,
            ),
            "to_thread": patch("asyncio.to_thread", side_effect=lambda fn: fn()),
            "db_retry": patch(
                "app.db_resilience.run_with_db_retry",
                side_effect=lambda fn, **kw: fn(),
            ),
            "open_ids": patch(
                "app.services.ctrader_client.get_open_position_ids_for_user_with_retry",
                new_callable=AsyncMock,
                return_value=set(),
            ),
            "deal": patch(
                "app.services.ctrader_client.get_position_close_detail_for_user",
                new_callable=AsyncMock,
                return_value=None,
            ),
            "close": patch(
                "app.services.strategy_executor._close_live_forex_execution_with_db_retry",
                new_callable=AsyncMock,
                return_value=True,
            ),
            "audit": patch(
                "app.services.trade_management.reconcile_broker_pnl_for_recent_closes",
                new_callable=AsyncMock,
                return_value={},
            ),
            "orphan": patch(
                "app.services.strategy_heal.expire_untracked_forex_opens_when_broker_empty",
                new_callable=AsyncMock,
            ),
        }
        patches.update(extra)
        return patches

    async def test_estimated_close_after_timeout(self):
        work = [{
            "exec_id": 6803,
            "user_id": 9,
            "user": MagicMock(id=9),
            "position_id": 144726466,
            "symbol": "XAUUSD",
            "direction": "LONG",
            "entry": 2650.0,
            "tp_price": 2660.0,
            "sl_price": 2640.0,
            "be_moved": False,
            "ctrader_account_id": "12345",
            "notes": "pos=144726466 | acct=12345",
            "fired_at": None,
        }]
        _FX_RECONCILE_MISSING[6803] = 2
        _FX_RECONCILE_MISSING_SINCE[6803] = time.monotonic() - _FX_RECONCILE_ESTIMATE_AFTER_S - 1

        p = self._patch_reconcile(work)
        with p["worklist"], p["to_thread"], p["db_retry"], p["open_ids"], p["deal"], \
                p["audit"], p["orphan"], \
                patch(
                    "app.services.strategy_executor._estimate_reconcile_exit",
                    new_callable=AsyncMock,
                    return_value=("LOSS", 2645.0),
                ) as mock_est, p["close"] as mock_close:
            await _reconcile_forex_closes()

        mock_est.assert_awaited_once()
        mock_close.assert_awaited_once_with(
            6803,
            "LOSS",
            2645.0,
            source="ctrader-reconcile-estimated",
            note_suffix="exit estimated — broker deal unavailable",
        )
        self.assertNotIn(6803, _FX_RECONCILE_MISSING)

    async def test_broker_deal_closes_immediately(self):
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
            "notes": "pos=777002",
            "fired_at": None,
        }]
        p = self._patch_reconcile(work)
        with p["worklist"], p["to_thread"], p["db_retry"], \
                patch(
                    "app.services.ctrader_client.get_open_position_ids_for_user_with_retry",
                    new_callable=AsyncMock,
                    return_value=set(),
                ), patch(
                    "app.services.ctrader_client.get_position_close_detail_for_user",
                    new_callable=AsyncMock,
                    return_value={"exit_price": 1.09, "outcome": "LOSS"},
                ), p["audit"], p["orphan"], p["close"] as mock_close:
            await _reconcile_forex_closes()

        mock_close.assert_awaited_once()
        self.assertEqual(mock_close.await_args[0][0], 502)

    async def test_clear_missing_resets_both_counters(self):
        _FX_RECONCILE_MISSING[1] = 2
        _FX_RECONCILE_MISSING_SINCE[1] = time.monotonic()
        _clear_reconcile_missing(1)
        self.assertNotIn(1, _FX_RECONCILE_MISSING)
        self.assertNotIn(1, _FX_RECONCILE_MISSING_SINCE)


if __name__ == "__main__":
    unittest.main()
