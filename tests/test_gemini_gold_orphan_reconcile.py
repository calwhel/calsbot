"""Orphan OPEN execution reconcile for gemini-gold."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.reconcile import reconcile_orphan_open_executions_sync
from app.strategy_models import StrategyExecution


def _make_ex(**kwargs):
    defaults = dict(
        id=1,
        user_id=42,
        symbol="XAUUSD",
        outcome="OPEN",
        entry_price=3986.0,
        ctrader_account_id="47664720",
        ctrader_order_id="",
        ctrader_position_id="",
        notes="gemini_gold_trader decision_id=9",
    )
    defaults.update(kwargs)
    return StrategyExecution(**defaults)


def test_keeps_live_mirror_open_when_demo_broker_flat():
    """Live-mirror OPEN rows must not be cancelled against the demo account poll."""
    db = MagicMock()
    demo_ex = _make_ex(id=201, notes="gemini_gold_trader decision_id=9")
    live_ex = _make_ex(
        id=202,
        ctrader_account_id="999888",
        notes="gemini_gold_trader_live_mirror decision_id=9 demo_exec=201",
    )
    # Query filter should exclude live_mirror — simulate by returning only demo rows
    # that match the updated SQL filter. Also assert the filter excludes live_mirror.
    from app.gemini_gold_trader import reconcile as recon

    captured = {}

    class _FakeQuery:
        def filter(self, *args, **kwargs):
            captured["filter_args"] = args
            return self

        def order_by(self, *args, **kwargs):
            return self

        def all(self):
            return [demo_ex]

    db.query.return_value = _FakeQuery()

    before, closed = recon.reconcile_orphan_open_executions_sync(
        db,
        user_id=42,
        demo_ctid="47664720",
        broker_open_position_ids=set(),
        dry_run=False,
    )

    assert len(before) == 1
    assert len(closed) == 1
    assert demo_ex.outcome == "CANCELLED"
    assert live_ex.outcome == "OPEN"
    filter_src = " ".join(str(a) for a in captured.get("filter_args", ()))
    assert "NOT LIKE" in filter_src.upper()


def test_cancels_open_row_without_broker_position_id():
    db = MagicMock()
    ex = _make_ex(id=101)
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [ex]

    before, closed = reconcile_orphan_open_executions_sync(
        db,
        user_id=42,
        demo_ctid="47664720",
        broker_open_position_ids=set(),
        dry_run=False,
    )

    assert len(before) == 1
    assert len(closed) == 1
    assert ex.outcome == "CANCELLED"
    assert ex.closed_at is not None
    assert "broker flat on demo ctid" in (ex.notes or "")
    db.commit.assert_called()


def test_keeps_row_when_broker_position_still_open():
    db = MagicMock()
    ex = _make_ex(id=102, ctrader_position_id="555001", notes="gemini_gold_trader | pos=555001")
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [ex]

    before, closed = reconcile_orphan_open_executions_sync(
        db,
        user_id=42,
        demo_ctid="47664720",
        broker_open_position_ids={555001},
        dry_run=False,
    )

    assert len(before) == 1
    assert closed == []
    assert ex.outcome == "OPEN"


def test_cancels_row_when_broker_position_missing():
    db = MagicMock()
    ex = _make_ex(id=103, ctrader_position_id="555002", notes="gemini_gold_trader | pos=555002")
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [ex]

    before, closed = reconcile_orphan_open_executions_sync(
        db,
        user_id=42,
        demo_ctid="47664720",
        broker_open_position_ids=set(),
        dry_run=False,
    )

    assert len(closed) == 1
    assert ex.outcome == "CANCELLED"
    assert "broker flat on demo ctid" in (ex.notes or "")


def test_cancels_all_open_rows_when_broker_flat():
    db = MagicMock()
    ex1 = _make_ex(id=105, ctrader_position_id="555003")
    ex2 = _make_ex(id=106, ctrader_account_id="47516246", ctrader_position_id="555004")
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [ex1, ex2]

    before, closed = reconcile_orphan_open_executions_sync(
        db,
        user_id=42,
        demo_ctid="47664720",
        broker_open_position_ids=set(),
        dry_run=False,
    )

    assert len(before) == 2
    assert len(closed) == 2
    assert ex1.outcome == "CANCELLED"
    assert ex2.outcome == "CANCELLED"


def test_dry_run_lists_without_mutating():
    db = MagicMock()
    ex = _make_ex(id=104)
    db.query.return_value.filter.return_value.order_by.return_value.all.return_value = [ex]

    before, closed = reconcile_orphan_open_executions_sync(
        db,
        user_id=42,
        demo_ctid="47664720",
        broker_open_position_ids=set(),
        dry_run=True,
    )

    assert len(before) == 1
    assert len(closed) == 1
    assert ex.outcome == "OPEN"
    db.commit.assert_not_called()
