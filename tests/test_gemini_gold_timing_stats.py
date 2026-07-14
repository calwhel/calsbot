"""Gemini Gold UTC hour performance stats (cTrader executions)."""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.gemini_gold_trader.timing_stats import hour_performance_stats


def _ex(*, hour: int, outcome: str, decision_id: int = 1, ctid: str = "12345"):
    fired = datetime.utcnow().replace(hour=hour, minute=15, second=0, microsecond=0)
    return SimpleNamespace(
        fired_at=fired,
        closed_at=fired + timedelta(hours=1),
        outcome=outcome,
        ctrader_account_id=ctid,
        notes=f"gemini_gold_trader decision_id={decision_id}",
        conditions_met=None,
    )


@patch("app.gemini_gold_trader.timing_stats._load_decision_meta", return_value={})
@patch("app.gemini_gold_trader.timing_stats.gemini_broker_executions_query")
def test_hour_performance_stats_from_ctrader_executions(mock_query, _mock_meta):
    closed = [
        _ex(hour=8, outcome="WIN", decision_id=1),
        _ex(hour=8, outcome="WIN", decision_id=2),
        _ex(hour=8, outcome="LOSS", decision_id=3),
        _ex(hour=14, outcome="LOSS", decision_id=4),
        _ex(hour=14, outcome="LOSS", decision_id=5),
    ]
    closed_q = MagicMock()
    closed_q.all.return_value = closed
    fired_q = MagicMock()
    fired_q.all.return_value = closed
    mock_query.side_effect = [closed_q, fired_q]

    stats = hour_performance_stats(
        MagicMock(),
        user_id=42,
        ctrader_account_id="12345",
        days=14,
        min_trades=2,
    )
    assert stats["source"] == "ctrader_executions"
    assert stats["total_closed_trades"] == 5
    assert stats["overall_win_rate_pct"] == pytest.approx(40.0, abs=0.1)
    assert stats["best_hours"][0]["hour_utc"] == 8
    assert stats["best_hours"][0]["win_rate_pct"] == pytest.approx(66.7, abs=0.1)
    assert stats["worst_hours"][0]["hour_utc"] == 14


def test_hour_performance_stats_empty_without_user():
    stats = hour_performance_stats(MagicMock(), days=14)
    assert stats["total_closed_trades"] == 0
    assert stats["by_hour"] == []
