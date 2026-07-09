"""Unified cTrader reconcile snapshot — balance, equity, positions in one call."""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ctrader_client import (
    CTRADER_HOST_DEMO,
    _parse_reconcile_position_ids,
    _parse_reconcile_snapshot,
    get_account_reconcile_resilient,
    get_broker_reconcile_snapshot_resilient,
)


class _FakePosition:
    def __init__(self, pid):
        self.positionId = pid


class _FakeReconcileRes:
    def __init__(self, *, balance=None, equity=None, positions=None):
        self.balance = balance
        self.equity = equity
        self.position = positions or []


class TestParseReconcileSnapshot(unittest.TestCase):
    def test_parses_balance_equity_and_positions(self):
        res = _FakeReconcileRes(balance=100000, equity=100500, positions=[_FakePosition(42)])
        snap = _parse_reconcile_snapshot(res)
        self.assertEqual(snap["balance"], 1000.0)
        self.assertEqual(snap["equity"], 1005.0)
        self.assertEqual(snap["position_ids"], {42})

    def test_empty_positions_is_flat_set(self):
        res = _FakeReconcileRes(balance=50000)
        self.assertEqual(_parse_reconcile_position_ids(res), set())


class TestBrokerReconcileSnapshotResilient(unittest.TestCase):
    def test_demo_host_routing(self):
        prefs = SimpleNamespace(
            ctrader_accounts='[{"ctidTraderAccountId":47782488,"isLive":false}]'
        )
        with patch(
            "app.services.ctrader_client._get_broker_reconcile_snapshot",
            new_callable=AsyncMock,
            return_value={
                "balance": 2500.0,
                "equity": 2500.0,
                "position_ids": set(),
                "host": CTRADER_HOST_DEMO,
            },
        ) as mock_snap:
            out = asyncio.run(
                get_broker_reconcile_snapshot_resilient(
                    "tok",
                    47782488,
                    prefs=prefs,
                    user_id=1,
                )
            )
        self.assertIsNotNone(out["position_ids"])
        self.assertEqual(out["balance"], 2500.0)
        mock_snap.assert_awaited()
        self.assertEqual(mock_snap.await_args.kwargs.get("host"), CTRADER_HOST_DEMO)

    def test_unreachable_returns_error(self):
        prefs = SimpleNamespace(ctrader_accounts="[]")
        with patch(
            "app.services.ctrader_client._get_broker_reconcile_snapshot",
            new_callable=AsyncMock,
            return_value=None,
        ), patch(
            "app.services.ctrader_client._latest_ctrader_access_token",
            return_value=None,
        ):
            out = asyncio.run(
                get_broker_reconcile_snapshot_resilient("tok", 1, prefs=prefs)
            )
        self.assertIsNone(out["position_ids"])
        self.assertEqual(out["error"], "broker_unreachable")

    def test_account_reconcile_resilient_delegates(self):
        with patch(
            "app.services.ctrader_client.get_broker_reconcile_snapshot_resilient",
            new_callable=AsyncMock,
            return_value={
                "balance": 100.0,
                "equity": 101.0,
                "position_ids": set(),
                "error": None,
                "host": CTRADER_HOST_DEMO,
            },
        ):
            acct = asyncio.run(get_account_reconcile_resilient("tok", 1))
        self.assertEqual(acct, {"balance": 100.0, "equity": 101.0})


class TestFetchCtraderAccountSnapshot(unittest.TestCase):
    def test_single_reconcile_populates_broker_fields(self):
        from app.gemini_gold_trader.config import env_defaults
        from app.gemini_gold_trader.guardrails import merge_config
        from app.gemini_gold_trader.models import GeminiGoldConfig
        from app.gemini_gold_trader.review import _fetch_ctrader_account_snapshot

        row = GeminiGoldConfig(
            id=1,
            enabled=True,
            dry_run=True,
            demo_ctrader_account_id="47782488",
        )
        cfg = merge_config(row, env_defaults())
        db = MagicMock()
        user = SimpleNamespace(id=7, ctrader_access_token="tok")
        prefs = SimpleNamespace(
            ctrader_access_token="tok",
            ctrader_accounts='[{"ctidTraderAccountId":47782488,"isLive":false,"balance":3000}]',
        )
        db.query.return_value.filter.return_value.first.side_effect = [user, prefs]

        async def _run():
            with patch(
                "app.gemini_gold_trader.reconcile.list_open_executions",
                return_value=[],
            ), patch(
                "app.services.ctrader_client.get_broker_reconcile_snapshot_resilient",
                new_callable=AsyncMock,
                return_value={
                    "balance": 3000.0,
                    "equity": 3005.0,
                    "position_ids": {99},
                    "error": None,
                    "host": CTRADER_HOST_DEMO,
                },
            ):
                return await _fetch_ctrader_account_snapshot(
                    db, user_id=7, cfg=cfg, days=14
                )

        snap = asyncio.run(_run())
        self.assertEqual(snap["balance"], 3000.0)
        self.assertEqual(snap["broker_open_position_count"], 1)
        self.assertEqual(snap["position_reconciliation"], "mismatch")
        self.assertFalse(snap["broker_unreachable"])


if __name__ == "__main__":
    unittest.main()
