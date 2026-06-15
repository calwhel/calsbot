"""Multi-account cTrader routing + parallel fan-out."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ctrader_client import (
    list_added_ctrader_ctids,
    normalize_account_lot,
    parse_added_accounts_json,
    resolve_ctrader_ctid,
)
from app.services.ctrader_order_queue import CtraderOrderJob, CtraderSlAmendJob
from app.strategy_models import StrategyAccountAssignment, UserStrategy


class TestAddedAccounts(unittest.TestCase):
    def test_parse_added_accounts_json(self):
        self.assertEqual(parse_added_accounts_json('["1","2"]'), ["1", "2"])
        self.assertEqual(parse_added_accounts_json(None), [])

    def test_list_assignable_includes_added_and_default(self):
        prefs = SimpleNamespace(
            ctrader_added_accounts='["222","333"]',
            ctrader_account_id="111",
        )
        from app.services.ctrader_client import list_assignable_ctrader_ctids
        self.assertEqual(
            list_assignable_ctrader_ctids(prefs),
            ["111", "222", "333"],
        )


class TestAssignmentTable(unittest.TestCase):
    def test_strategy_account_assignments_table_exists(self):
        col = StrategyAccountAssignment.__table__.columns.get("lot_size")
        self.assertIsNotNone(col)
        self.assertTrue(col.nullable)
        self.assertIsNotNone(StrategyAccountAssignment.__table__.columns.get("enabled"))
        self.assertIsNotNone(
            StrategyAccountAssignment.__table__.columns.get("ctrader_account_id")
        )
        self.assertIsNone(StrategyAccountAssignment.__table__.columns.get("ctid"))

    def test_normalize_account_lot(self):
        self.assertEqual(normalize_account_lot(0.25), 0.25)
        self.assertEqual(normalize_account_lot(0.025), 0.03)
        self.assertIsNone(normalize_account_lot(0))


class TestParallelFanOut(unittest.TestCase):
    def test_fire_path_has_fanout_gather(self):
        from app.services import strategy_executor as se
        src = inspect.getsource(se)
        self.assertIn("_ctrader_fanout_live_fire", src)
        self.assertIn("asyncio.gather", src)
        self.assertIn("signal_group_id", src)
        self.assertIn("get_enabled_fire_targets", src)

    def test_fanout_creates_multiple_jobs(self):
        from app.services import strategy_executor as se
        src = inspect.getsource(se._ctrader_fanout_live_fire)
        self.assertIn("CtraderOrderJob(", src)
        self.assertIn("enqueue_ctrader_order", src)
        self.assertIn("signal_group_id=_signal_group_id", src)

    def test_order_worker_spawns_concurrent_tasks(self):
        src = inspect.getsource(
            __import__(
                "app.services.ctrader_order_queue",
                fromlist=["_ctrader_order_worker"],
            )._ctrader_order_worker
        )
        self.assertIn("asyncio.create_task(_run_order_job(job))", src)


class TestPerStrategyApi(unittest.TestCase):
    def test_assignment_endpoints_exist(self):
        import strategy_portal_server as sps
        routes = [getattr(r, "path", "") for r in sps.app.routes]
        self.assertIn("/api/strategies/{strategy_id}/account-assignments", routes)
        self.assertIn("/api/live-forex/add-account", routes)


class TestResolveCtraderCtid(unittest.TestCase):
    def test_strategy_binding_before_prefs_default(self):
        ctid = resolve_ctrader_ctid(
            strategy_account_id="333",
            prefs_default="444",
        )
        self.assertEqual(ctid, "333")


class TestSchemaAndJobFields(unittest.TestCase):
    def test_order_job_carries_ctrader_account_id(self):
        job = CtraderOrderJob(
            user_id=1,
            strategy_id=2,
            execution_id=3,
            symbol="EURUSD",
            direction="LONG",
            entry_price=1.1,
            tp_pct=0.5,
            sl_pct=0.25,
            ctrader_account_id="55555",
            fixed_lots=0.01,
        )
        self.assertEqual(job.ctrader_account_id, "55555")
        self.assertEqual(job.fixed_lots, 0.01)


class TestFireTargetsHelper(unittest.TestCase):
    def test_enabled_assignments_from_table(self):
        from app.services.strategy_account_assignments import get_enabled_fire_targets

        strategy = SimpleNamespace(id=1, ctrader_account_id=None, ctrader_account_lot=None)
        prefs = SimpleNamespace(ctrader_account_id="999")
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [
            SimpleNamespace(ctrader_account_id="111", lot_size=0.01, enabled=True),
            SimpleNamespace(ctrader_account_id="222", lot_size=0.25, enabled=True),
        ]
        targets = get_enabled_fire_targets(mock_db, strategy, prefs)
        self.assertEqual(len(targets), 2)
        self.assertEqual(targets[0]["ctrader_account_id"], "111")
        self.assertEqual(targets[1]["lot_size"], 0.25)


if __name__ == "__main__":
    unittest.main()
