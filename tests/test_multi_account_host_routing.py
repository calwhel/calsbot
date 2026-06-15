"""Per-ctid demo/live host routing for orders, balance, reconcile."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.services.ctrader_client import (
    CTRADER_HOST_DEMO,
    CTRADER_HOST_LIVE,
    _routing_hosts_for_account,
    _should_try_alternate_order_host,
    place_market_order_resilient,
)


class TestRoutingHosts(unittest.TestCase):
    def test_demo_ctid_demo_host_only(self):
        prefs = SimpleNamespace(
            ctrader_accounts='[{"ctidTraderAccountId":47516246,"isLive":false}]'
        )
        self.assertEqual(_routing_hosts_for_account(prefs, 47516246), [CTRADER_HOST_DEMO])

    def test_live_ctid_live_host_only(self):
        prefs = SimpleNamespace(
            ctrader_accounts='[{"ctidTraderAccountId":47465772,"isLive":true}]'
        )
        self.assertEqual(_routing_hosts_for_account(prefs, 47465772), [CTRADER_HOST_LIVE])

    def test_unknown_ctid_tries_both_hosts(self):
        prefs = SimpleNamespace(ctrader_accounts="[]")
        hosts = _routing_hosts_for_account(prefs, 99999)
        self.assertEqual(hosts[0], CTRADER_HOST_LIVE)
        self.assertIn(CTRADER_HOST_DEMO, hosts)

    def test_alternate_host_on_cancel_when_type_unknown(self):
        self.assertTrue(
            _should_try_alternate_order_host("ORDER_CANCELLED", known_account_type=False)
        )
        self.assertFalse(
            _should_try_alternate_order_host("ORDER_CANCELLED", known_account_type=True)
        )


class TestOrderHostFallback(unittest.TestCase):
    def test_demo_order_retries_on_demo_after_live_cancel(self):
        prefs = SimpleNamespace(ctrader_accounts="[]")
        live_cancel = {
            "order_id": None,
            "actual_fill": None,
            "error": "ORDER_CANCELLED",
            "volume": 100000,
        }
        demo_fill = {
            "order_id": "1",
            "actual_fill": 1.085,
            "position_id": "99",
            "volume": 100000,
            "error": None,
        }
        with patch(
            "app.services.ctrader_client.place_order",
            new_callable=AsyncMock,
            side_effect=[live_cancel, demo_fill],
        ) as mock_place, patch(
            "app.services.ctrader_client._persist_account_host_metadata",
        ) as mock_persist:
            result = asyncio.run(
                place_market_order_resilient(
                    user_id=1,
                    access_token="tok",
                    ctid=47516246,
                    prefs=prefs,
                    symbol_name="EURUSD",
                    direction="LONG",
                    volume_lots=1.0,
                    entry_price=1.085,
                    sl_pct=0.25,
                    tp_pct=0.5,
                    execution_id=42,
                )
            )
        self.assertEqual(result.get("actual_fill"), 1.085)
        self.assertEqual(mock_place.await_count, 2)
        self.assertEqual(mock_place.await_args_list[0].kwargs.get("host"), CTRADER_HOST_LIVE)
        self.assertEqual(mock_place.await_args_list[1].kwargs.get("host"), CTRADER_HOST_DEMO)
        mock_persist.assert_called_once_with(1, 47516246, CTRADER_HOST_DEMO)


if __name__ == "__main__":
    unittest.main()
