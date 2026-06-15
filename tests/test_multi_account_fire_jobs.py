"""Strategy fire builds one order per enabled account with per-ctid routing."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import asyncio
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ctrader_client import CTRADER_HOST_DEMO, CTRADER_HOST_LIVE
from app.services.strategy_account_assignments import resolve_fire_target_routing


class TestResolveFireTargetRouting(unittest.TestCase):
    def test_demo_and_live_get_different_hosts(self):
        prefs = SimpleNamespace(
            ctrader_accounts=(
                '[{"ctidTraderAccountId":47516246,"isLive":false},'
                '{"ctidTraderAccountId":47465772,"isLive":true}]'
            )
        )
        demo = resolve_fire_target_routing(
            prefs, {"ctrader_account_id": "47516246", "lot_size": 0.05}
        )
        live = resolve_fire_target_routing(
            prefs, {"ctrader_account_id": "47465772", "lot_size": 0.10}
        )
        self.assertEqual(demo["hosts"], [CTRADER_HOST_DEMO])
        self.assertEqual(live["hosts"], [CTRADER_HOST_LIVE])
        self.assertEqual(demo["fixed_lots"], 0.05)
        self.assertEqual(live["fixed_lots"], 0.10)
        self.assertFalse(demo["is_live"])
        self.assertTrue(live["is_live"])


class TestFanOutJobBuilding(unittest.TestCase):
    def test_two_enabled_accounts_build_two_jobs(self):
        from app.services import strategy_executor as se
        from app.services.ctrader_order_queue import CtraderOrderJob

        prefs = SimpleNamespace(
            ctrader_accounts=(
                '[{"ctidTraderAccountId":47516246,"isLive":false},'
                '{"ctidTraderAccountId":47465772,"isLive":true}]'
            )
        )
        fire_targets = [
            {"ctrader_account_id": "47516246", "lot_size": 0.05},
            {"ctrader_account_id": "47465772", "lot_size": 0.10},
        ]
        jobs = []
        for target in fire_targets:
            routing = resolve_fire_target_routing(prefs, target)
            self.assertIsNotNone(routing)
            jobs.append(CtraderOrderJob(
                user_id=1,
                strategy_id=9,
                execution_id=100 + len(jobs),
                symbol="EURUSD",
                direction="LONG",
                entry_price=1.1,
                tp_pct=0.5,
                sl_pct=0.25,
                ctrader_account_id=routing["ctrader_account_id"],
                ctrader_hosts=tuple(routing["hosts"]),
                account_is_live=routing.get("is_live"),
                fixed_lots=routing["fixed_lots"],
            ))

        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].ctrader_account_id, "47516246")
        self.assertEqual(jobs[1].ctrader_account_id, "47465772")
        self.assertEqual(jobs[0].ctrader_hosts, (CTRADER_HOST_DEMO,))
        self.assertEqual(jobs[1].ctrader_hosts, (CTRADER_HOST_LIVE,))
        self.assertEqual(jobs[0].fixed_lots, 0.05)
        self.assertEqual(jobs[1].fixed_lots, 0.10)

    def test_fire_path_uses_multi_account_for_one_or_more(self):
        from app.services import strategy_executor as se

        src = inspect.getsource(se.evaluate_and_fire)
        self.assertIn("len(_fire_targets) >= 1", src)
        self.assertIn("resolve_fire_target_routing", inspect.getsource(se._ctrader_fanout_live_fire))


class TestOrderLogFormat(unittest.TestCase):
    def test_log_lines_demo_and_live(self):
        from app.services.ctrader_client import _log_order_route
        import logging

        records = []

        class H(logging.Handler):
            def emit(self, record):
                records.append(record.getMessage())

        h = H()
        logger = logging.getLogger("app.services.ctrader_client")
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        try:
            _log_order_route(
                execution_id=101,
                ctid=47516246,
                host=CTRADER_HOST_DEMO,
                result={"actual_fill": 1.085},
                lots=0.05,
                is_live=False,
            )
            _log_order_route(
                execution_id=102,
                ctid=47465772,
                host=CTRADER_HOST_LIVE,
                result={"actual_fill": 1.085},
                lots=0.10,
                is_live=True,
            )
        finally:
            logger.removeHandler(h)

        self.assertEqual(len(records), 2)
        self.assertIn(
            "[order] exec=101 ctid=47516246 is_live=False host=demo.ctraderapi.com lots=0.05 → fill",
            records[0],
        )
        self.assertIn(
            "[order] exec=102 ctid=47465772 is_live=True host=live.ctraderapi.com lots=0.1 → fill",
            records[1],
        )


if __name__ == "__main__":
    unittest.main()
