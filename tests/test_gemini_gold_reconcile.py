"""Tests for gemini-gold open execution diagnostics."""
from __future__ import annotations

import os
from unittest.mock import MagicMock

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.reconcile import list_open_executions


def test_list_open_executions_returns_execution_summary():
    ex = MagicMock()
    ex.id = 99
    ex.fired_at = None
    ex.direction = "LONG"
    ex.entry_price = 2650.0
    ex.ctrader_order_id = "111"
    ex.ctrader_position_id = "222"
    ex.notes = "gemini_gold_trader decision_id=7"

    db = MagicMock()
    q = db.query.return_value.filter.return_value.order_by.return_value
    q.all.return_value = [ex]

    rows = list_open_executions(db, 42)
    assert len(rows) == 1
    assert rows[0]["execution_id"] == 99
    assert "gemini_gold" in rows[0]["notes"]
