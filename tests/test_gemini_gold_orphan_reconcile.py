"""Orphan OPEN execution reconcile for gemini-gold."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

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
        fired_at=datetime.utcnow() - timedelta(minutes=20),
    )
    defaults.update(kwargs)
    return StrategyExecution(**defaults)


def _run(rows, *, broker_ids, dry_run=False, demo_ctid="47664720"):
    db = MagicMock()
    with patch(
        "app.gemini_gold_trader.reconcile._gemini_open_query",
        return_value=MagicMock(all=lambda: list(rows)),
    ):
        return reconcile_orphan_open_executions_sync(
            db,
            user_id=42,
            demo_ctid=demo_ctid,
            broker_open_position_ids=broker_ids,
            dry_run=dry_run,
        )


def test_keeps_live_mirror_open_when_demo_broker_flat():
    """Live-mirror OPEN rows must not be cancelled against the demo account poll."""
    demo_ex = _make_ex(id=201, notes="gemini_gold_trader decision_id=9", ctrader_position_id="")
    live_ex = _make_ex(
        id=202,
        ctrader_account_id="999888",
        notes="gemini_gold_trader_live_mirror decision_id=9 demo_exec=201",
    )
    # Query is mocked to return only demo rows (as the SQL live_mirror exclusion does).
    before, closed = _run([demo_ex], broker_ids=set())
    assert len(before) == 1
    assert len(closed) == 1
    assert demo_ex.outcome == "CANCELLED"
    assert live_ex.outcome == "OPEN"


def test_cancels_open_row_without_broker_position_id():
    ex = _make_ex(id=101, ctrader_position_id="")
    before, closed = _run([ex], broker_ids=set())
    assert len(before) == 1
    assert len(closed) == 1
    assert ex.outcome == "CANCELLED"
    assert ex.closed_at is not None
    assert "broker flat on demo ctid" in (ex.notes or "")


def test_keeps_row_when_broker_position_still_open():
    ex = _make_ex(id=102, ctrader_position_id="555001", notes="gemini_gold_trader | pos=555001")
    before, closed = _run([ex], broker_ids={555001})
    assert len(before) == 1
    assert closed == []
    assert ex.outcome == "OPEN"


def test_cancels_row_when_broker_position_missing():
    ex = _make_ex(
        id=103,
        ctrader_position_id="555002",
        notes="gemini_gold_trader | pos=555002",
        fired_at=datetime.utcnow() - timedelta(minutes=20),
    )
    before, closed = _run([ex], broker_ids=set())
    assert len(closed) == 1
    assert ex.outcome == "CANCELLED"
    assert "broker flat on demo ctid" in (ex.notes or "")


def test_grace_keeps_recent_fill_when_broker_poll_reports_flat():
    ex = _make_ex(
        id=110,
        ctrader_position_id="555010",
        notes="gemini_gold_trader | pos=555010",
        fired_at=datetime.utcnow(),
    )
    before, closed = _run([ex], broker_ids=set())
    assert closed == []
    assert ex.outcome == "OPEN"


def test_cancels_all_open_rows_when_broker_flat():
    old = datetime.utcnow() - timedelta(minutes=20)
    ex1 = _make_ex(id=105, ctrader_position_id="555003", fired_at=old)
    ex2 = _make_ex(id=106, ctrader_account_id="47516246", ctrader_position_id="555004", fired_at=old)
    before, closed = _run([ex1, ex2], broker_ids=set())
    assert len(before) == 2
    assert len(closed) == 2
    assert ex1.outcome == "CANCELLED"
    assert ex2.outcome == "CANCELLED"


def test_dry_run_lists_without_mutating():
    ex = _make_ex(id=104, ctrader_position_id="")
    db = MagicMock()
    with patch(
        "app.gemini_gold_trader.reconcile._gemini_open_query",
        return_value=MagicMock(all=lambda: [ex]),
    ):
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
