"""cTrader balance fetch — host selection, cache TTL, rate-limited warnings."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import time
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ctrader_client import (
    BALANCE_CONNECT_TIMEOUT_S,
    BALANCE_REQ_TIMEOUT_S,
    CTRADER_HOST_DEMO,
    CTRADER_HOST_LIVE,
    _BALANCE_CACHE_TTL,
    _balance_hosts_for_account,
    get_account_balance_resilient,
)


class TestBalanceHostSelection(unittest.TestCase):
    def test_demo_account_uses_demo_host_only(self):
        prefs = SimpleNamespace(ctrader_accounts='[{"ctidTraderAccountId":47465772,"isLive":false}]')
        hosts = _balance_hosts_for_account(prefs, 47465772)
        self.assertEqual(hosts, [CTRADER_HOST_DEMO])

    def test_live_account_uses_live_host_only(self):
        prefs = SimpleNamespace(ctrader_accounts='[{"ctidTraderAccountId":47516246,"isLive":true}]')
        hosts = _balance_hosts_for_account(prefs, 47516246)
        self.assertEqual(hosts, [CTRADER_HOST_LIVE])

    def test_unknown_account_tries_primary_then_alternate(self):
        prefs = SimpleNamespace(ctrader_accounts="[]")
        hosts = _balance_hosts_for_account(prefs, 99999)
        self.assertEqual(hosts[0], CTRADER_HOST_LIVE)
        self.assertEqual(len(hosts), 2)
        self.assertIn(CTRADER_HOST_DEMO, hosts)


class TestBalanceCacheAndTimeouts(unittest.TestCase):
    def test_balance_cache_ttl_is_sixty_seconds(self):
        self.assertEqual(_BALANCE_CACHE_TTL, 60.0)

    def test_balance_connect_and_req_timeouts_are_ten_seconds(self):
        self.assertEqual(BALANCE_CONNECT_TIMEOUT_S, 10.0)
        self.assertEqual(BALANCE_REQ_TIMEOUT_S, 10.0)

    def test_resilient_uses_host_helper(self):
        prefs = SimpleNamespace(ctrader_accounts='[{"ctidTraderAccountId":1,"isLive":false}]')
        with patch(
            "app.services.ctrader_client._get_account_balance",
            new_callable=AsyncMock,
            return_value=1234.56,
        ) as mock_bal:
            import asyncio
            bal = asyncio.run(
                get_account_balance_resilient("tok", 1, prefs=prefs),
            )
        self.assertEqual(bal, 1234.56)
        mock_bal.assert_awaited_once()
        self.assertEqual(mock_bal.await_args.kwargs.get("host"), CTRADER_HOST_DEMO)


class TestBalanceWarnRateLimit(unittest.TestCase):
    def test_warn_once_per_ctid_per_minute(self):
        import strategy_portal_server as sps
        sps._lf_balance_warn_at.clear()
        with patch.object(sps.logger, "warning") as mock_warn:
            sps._lf_log_balance_fetch_warn("UID1", "47465772", "TimeoutError")
            sps._lf_log_balance_fetch_warn("UID1", "47465772", "TimeoutError")
            sps._lf_log_balance_fetch_warn("UID1", "47516246", "TimeoutError")
        self.assertEqual(mock_warn.call_count, 2)


if __name__ == "__main__":
    unittest.main()
