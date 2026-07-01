"""Broker close reconciliation helpers for Gemini Gold Trader."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def list_open_executions(db, user_id: int) -> List[Dict[str, Any]]:
    """OPEN gemini_gold StrategyExecution rows (used for cap diagnostics)."""
    from app.strategy_models import StrategyExecution

    rows = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.user_id == user_id,
            StrategyExecution.symbol == "XAUUSD",
            StrategyExecution.outcome == "OPEN",
            StrategyExecution.notes.like("%gemini_gold_trader%"),
        )
        .order_by(StrategyExecution.fired_at.desc())
        .all()
    )
    out: List[Dict[str, Any]] = []
    for ex in rows:
        out.append(
            {
                "execution_id": ex.id,
                "fired_at": ex.fired_at.isoformat() if ex.fired_at else None,
                "direction": ex.direction,
                "entry_price": float(ex.entry_price) if ex.entry_price else None,
                "ctrader_order_id": ex.ctrader_order_id,
                "ctrader_position_id": ex.ctrader_position_id,
                "notes": (ex.notes or "")[:240],
            }
        )
    return out
