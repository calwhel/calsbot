"""Post-trade outcome recording for Gemini Gold."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldOutcome

logger = logging.getLogger(__name__)


def compute_r_multiple(
    *,
    direction: str,
    entry: float,
    stop_loss: float,
    pnl_pct: float,
) -> Optional[float]:
    if entry <= 0 or stop_loss <= 0:
        return None
    risk = abs(entry - stop_loss) / entry * 100.0
    if risk <= 0:
        return None
    return float(pnl_pct) / risk


def record_outcome_from_execution(db, decision_id: int, execution) -> bool:
    if not execution or execution.outcome == "OPEN":
        return False
    existing = (
        db.query(GeminiGoldOutcome)
        .filter(GeminiGoldOutcome.decision_id == decision_id)
        .first()
    )
    if existing:
        return False

    dec = db.query(GeminiGoldDecision).filter(GeminiGoldDecision.id == decision_id).first()
    d = (dec.decision or {}) if dec else {}
    entry = float(d.get("entry") or execution.entry_price or 0)
    sl = float(d.get("stop_loss") or execution.sl_price or 0)
    direction = (d.get("direction") or execution.direction or "long").upper()

    result = "breakeven"
    if execution.outcome == "WIN":
        result = "win"
    elif execution.outcome == "LOSS":
        result = "loss"

    pnl = float(execution.pnl_pct or 0)
    r_mult = compute_r_multiple(
        direction=direction,
        entry=entry,
        stop_loss=sl,
        pnl_pct=pnl,
    )

    row = GeminiGoldOutcome(
        decision_id=decision_id,
        session=getattr(dec, "session", None) if dec else None,
        setup_type=str(d.get("setup_type") or getattr(dec, "setup_type", None) or "") or None,
        result=result,
        pnl=pnl,
        r_multiple=r_mult,
        closed_ts=execution.closed_at or datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    logger.info(
        "[gemini-gold] outcome recorded decision_id=%s result=%s pnl=%.2f%%",
        decision_id,
        result,
        pnl,
    )
    return True


def sync_closed_outcomes(db, user_id: int) -> int:
    """Scan recently closed gemini_gold executions and record outcomes."""
    from app.strategy_models import StrategyExecution

    rows = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.user_id == user_id,
            StrategyExecution.symbol == "XAUUSD",
            StrategyExecution.notes.like("%gemini_gold_trader%"),
            StrategyExecution.outcome.in_(("WIN", "LOSS", "BREAKEVEN")),
            StrategyExecution.closed_at.isnot(None),
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(20)
        .all()
    )
    recorded = 0
    for ex in rows:
        notes = ex.notes or ""
        decision_id = None
        if "decision_id=" in notes:
            try:
                decision_id = int(notes.split("decision_id=")[1].split()[0].strip("|"))
            except (ValueError, IndexError):
                pass
        if not decision_id and ex.conditions_met:
            try:
                decision_id = int((ex.conditions_met or {}).get("gemini_gold_decision_id"))
            except (TypeError, ValueError):
                pass
        if decision_id and record_outcome_from_execution(db, decision_id, ex):
            recorded += 1
    return recorded
