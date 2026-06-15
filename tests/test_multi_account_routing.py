"""Per-strategy cTrader account routing (no mirror fan-out)."""
import os
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret")

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.ctrader_client import (
    list_assignable_ctrader_ctids,
    parse_added_accounts_json,
    resolve_ctrader_ctid,
)
from app.services.ctrader_order_queue import CtraderOrderJob, CtraderSlAmendJob
from app.strategy_models import UserStrategy


class TestAddedAccounts(unittest.TestCase):
    def test_parse_added_accounts_json(self):
        self.assertEqual(parse_added_accounts_json('["1","2"]'), ["1", "2"])
        self.assertEqual(parse_added_accounts_json(None), [])

    def test_list_assignable_includes_added_and_default(self):
        prefs = SimpleNamespace(
            ctrader_added_accounts='["222","333"]',
            ctrader_account_id="111",
        )
        self.assertEqual(
            list_assignable_ctrader_ctids(prefs),
            ["111", "222", "333"],
        )

    def test_list_assignable_default_only_when_no_added(self):
        prefs = SimpleNamespace(
            ctrader_added_accounts=None,
            ctrader_account_id="999",
        )
        self.assertEqual(list_assignable_ctrader_ctids(prefs), ["999"])


class TestPerStrategyFirePath(unittest.TestCase):
    def test_fire_path_resolves_strategy_account_not_mirror(self):
        from app.services import strategy_executor as se
        src = inspect.getsource(se)
        self.assertIn("strategy_account_id=getattr(strategy, \"ctrader_account_id\", None)", src)
        self.assertNotIn("get_mirror_execution_ctids", src)
        self.assertNotIn("signal_group_id=", src)

    def test_single_ctrader_order_job_per_signal(self):
        from app.services import strategy_executor as se
        src = inspect.getsource(se)
        # One CtraderOrderJob enqueue per live fire — not a list/gather of mirror legs.
        self.assertIn("CtraderOrderJob(", src)
        self.assertIn("ctrader_account_id=_job_ctid", src)

    def test_order_worker_spawns_concurrent_tasks(self):
        src = inspect.getsource(
            __import__(
                "app.services.ctrader_order_queue",
                fromlist=["_ctrader_order_worker"],
            )._ctrader_order_worker
        )
        self.assertIn("asyncio.create_task(_run_order_job(job))", src)


class TestPerStrategyApi(unittest.TestCase):
    def test_add_and_assign_endpoints_exist(self):
        import strategy_portal_server as sps
        routes = [getattr(r, "path", "") for r in sps.app.routes]
        self.assertIn("/api/live-forex/add-account", routes)
        self.assertIn("/api/strategies/{strategy_id}/assign-account", routes)
        self.assertNotIn("/api/live-forex/mirror-account", routes)


class TestResolveCtraderCtid(unittest.TestCase):
    def test_execution_column_wins(self):
        ctid = resolve_ctrader_ctid(
            execution_account_id="111",
            notes="acct=222",
            strategy_account_id="333",
            prefs_default="444",
        )
        self.assertEqual(ctid, "111")

    def test_notes_acct_when_no_execution_column(self):
        ctid = resolve_ctrader_ctid(
            notes="pos=9 | acct=222",
            strategy_account_id="333",
            prefs_default="444",
        )
        self.assertEqual(ctid, "222")

    def test_strategy_binding_before_prefs_default(self):
        ctid = resolve_ctrader_ctid(
            strategy_account_id="333",
            prefs_default="444",
        )
        self.assertEqual(ctid, "333")

    def test_prefs_default_fallback(self):
        ctid = resolve_ctrader_ctid(prefs_default="444")
        self.assertEqual(ctid, "444")

    def test_null_strategy_uses_prefs(self):
        ctid = resolve_ctrader_ctid(
            strategy_account_id=None,
            prefs_default="999",
        )
        self.assertEqual(ctid, "999")


class TestSchemaAndJobFields(unittest.TestCase):
    def test_user_strategy_has_ctrader_account_id_column(self):
        col = UserStrategy.__table__.columns.get("ctrader_account_id")
        self.assertIsNotNone(col)
        self.assertTrue(col.nullable)

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
        )
        self.assertEqual(job.ctrader_account_id, "55555")

    def test_sl_amend_job_carries_ctrader_account_id(self):
        job = CtraderSlAmendJob(
            user_id=1,
            exec_id=2,
            position_id=99,
            new_sl=1.08,
            ctrader_account_id="66666",
        )
        self.assertEqual(job.ctrader_account_id, "66666")


class TestPlaceOrderExplicitCtid(unittest.IsolatedAsyncioTestCase):
    async def test_place_order_uses_explicit_ctid_not_prefs(self):
        from app.services.ctrader_client import place_ctrader_order_for_user

        user = SimpleNamespace(id=7)
        prefs = SimpleNamespace(
            ctrader_access_token="tok",
            ctrader_account_id="11111",
            ctrader_accounts=None,
        )

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = prefs

        captured = {}

        async def fake_resilient(**kwargs):
            captured.update(kwargs)
            return {"order_id": "oid", "actual_fill": 1.085, "position_id": "42"}

        with patch("app.database.SessionLocal", return_value=mock_db), \
             patch(
                 "app.services.ctrader_client.place_market_order_resilient",
                 new=AsyncMock(side_effect=fake_resilient),
             ), \
             patch(
                 "app.services.ctrader_client.compute_sltp_prices",
                 return_value=(1.09, 1.08),
             ), \
             patch("app.services.ctrader_client.validate_sltp_sanity", return_value=True):
            result = await place_ctrader_order_for_user(
                user=user,
                symbol="EURUSD",
                direction="LONG",
                entry_price=1.085,
                tp_pct=0.5,
                sl_pct=0.25,
                ctid="22222",
            )

        self.assertEqual(captured.get("ctid"), 22222)
        self.assertEqual(result.get("account_id"), "22222")


class TestAmendSlUsesExecutionAccount(unittest.IsolatedAsyncioTestCase):
    async def test_amend_sl_routes_to_execution_account(self):
        from app.services.ctrader_client import amend_position_sl_result

        user = SimpleNamespace(id=3)
        prefs = SimpleNamespace(
            ctrader_access_token="tok",
            ctrader_account_id="11111",
            ctrader_accounts='[{"ctidTraderAccountId":33333,"isLive":false}]',
        )
        execution = SimpleNamespace(
            ctrader_account_id="33333",
            notes="pos=1 | acct=33333",
        )

        def query_side(model):
            q = MagicMock()
            if model.__name__ == "User":
                q.filter.return_value.first.return_value = user
            elif model.__name__ == "UserPreference":
                q.filter.return_value.first.return_value = prefs
            elif model.__name__ == "StrategyExecution":
                q.filter.return_value.first.return_value = execution
            return q

        mock_db = MagicMock()
        mock_db.query.side_effect = query_side

        captured = {}

        async def fake_modify(access_token, ctid, position_id, **kwargs):
            captured["ctid"] = ctid
            return {"ok": True, "result": "confirmed", "broker_reply": {}}

        with patch("app.database.SessionLocal", return_value=mock_db), \
             patch(
                 "app.services.ctrader_client.modify_position_sltp_result",
                 new=AsyncMock(side_effect=fake_modify),
             ):
            res = await amend_position_sl_result(
                user_id=3,
                position_id=99,
                new_sl=1.08,
                exec_id=5,
            )

        self.assertTrue(res.get("ok"))
        self.assertEqual(captured.get("ctid"), 33333)


class TestReconcileGrouping(unittest.TestCase):
    def test_worklist_includes_ctrader_account_id_per_execution(self):
        from app.services.strategy_executor import _build_forex_reconcile_worklist

        ex1 = SimpleNamespace(
            id=1,
            user_id=10,
            strategy_id=100,
            notes="pos=101 | acct=11111",
            ctrader_account_id="11111",
            symbol="EURUSD",
            direction="LONG",
            entry_price=1.1,
            tp_price=1.11,
            tp2_price=None,
            sl_price=1.09,
        )
        ex2 = SimpleNamespace(
            id=2,
            user_id=10,
            strategy_id=101,
            notes="pos=102 | acct=22222",
            ctrader_account_id="22222",
            symbol="GBPUSD",
            direction="SHORT",
            entry_price=1.25,
            tp_price=1.24,
            tp2_price=None,
            sl_price=1.26,
        )

        user = SimpleNamespace(id=10)
        prefs = SimpleNamespace(ctrader_account_id="11111")
        strat1 = SimpleNamespace(ctrader_account_id=None)
        strat2 = SimpleNamespace(ctrader_account_id="22222")

        mock_db = MagicMock()

        def query_side(model):
            q = MagicMock()
            if model.__name__ == "StrategyExecution":
                q.filter.return_value.all.return_value = [ex1, ex2]
            elif model.__name__ == "User":
                q.filter.return_value.first.return_value = user
            elif model.__name__ == "UserPreference":
                q.filter.return_value.first.return_value = prefs
            elif model.__name__ == "UserStrategy":
                def strat_filter(*args, **kwargs):
                    sf = MagicMock()
                    def first():
                        if not hasattr(first, "_n"):
                            first._n = 0
                        first._n += 1
                        return strat1 if first._n == 1 else strat2
                    sf.filter.return_value.first = first
                    return sf
                q.filter = strat_filter
            return q

        mock_db.query.side_effect = query_side

        with patch("app.database.BgSessionLocal", return_value=mock_db), \
             patch(
                 "app.services.strategy_executor._ctrader_position_id_from_execution",
                 side_effect=lambda ex: 101 if ex.id == 1 else 102,
             ):
            work = _build_forex_reconcile_worklist()

        accts = {w["ctrader_account_id"] for w in work}
        self.assertEqual(accts, {"11111", "22222"})
        self.assertEqual(len(work), 2)

    def test_reconcile_groups_by_user_and_ctid(self):
        src = inspect.getsource(
            __import__(
                "app.services.strategy_executor",
                fromlist=["_reconcile_forex_closes"],
            )._reconcile_forex_closes
        )
        self.assertIn('key = (w["user_id"], w["ctrader_account_id"])', src)
        self.assertIn("get_open_position_ids_for_user(user_obj[uid], ctid=acct_id)", src)


if __name__ == "__main__":
    unittest.main()
