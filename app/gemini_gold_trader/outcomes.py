"""Post-trade outcome recording for Gemini Gold."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import or_

from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldOutcome

logger = logging.getLogger(__name__)

_GEMINI_BROKER_NOTES = "gemini_gold_trader decision_id=%"
_CLOSED_OUTCOMES: Tuple[str, ...] = ("WIN", "LOSS", "BREAKEVEN")


def decision_id_from_execution(ex) -> Optional[int]:
    notes = ex.notes or ""
    if "decision_id=" in notes:
        try:
            return int(notes.split("decision_id=")[1].split()[0].strip("|"))
        except (ValueError, IndexError):
            pass
    if ex.conditions_met:
        try:
            return int((ex.conditions_met or {}).get("gemini_gold_decision_id"))
        except (TypeError, ValueError):
            pass
    return None


def gemini_broker_executions_query(
    db,
    *,
    user_id: int,
    since: Optional[datetime] = None,
    ctrader_account_id: Optional[str] = None,
    closed_only: bool = False,
    outcomes: Optional[Sequence[str]] = None,
):
    """StrategyExecution rows for Gemini Gold on the configured cTrader account."""
    from app.strategy_models import StrategyExecution

    q = db.query(StrategyExecution).filter(
        StrategyExecution.user_id == int(user_id),
        StrategyExecution.symbol == "XAUUSD",
        StrategyExecution.notes.like(_GEMINI_BROKER_NOTES),
    )
    if closed_only:
        q = q.filter(
            StrategyExecution.closed_at.isnot(None),
            StrategyExecution.outcome.in_(tuple(outcomes or _CLOSED_OUTCOMES)),
        )
        if since is not None:
            q = q.filter(StrategyExecution.closed_at >= since)
    elif since is not None:
        q = q.filter(StrategyExecution.fired_at >= since)

    if ctrader_account_id:
        ctid = str(ctrader_account_id).strip()
        q = q.filter(
            or_(
                StrategyExecution.ctrader_account_id == ctid,
                StrategyExecution.ctrader_account_id.is_(None),
            )
        )
    return q


def broker_outcome_label(outcome: Optional[str]) -> str:
    raw = str(outcome or "").upper()
    if raw == "WIN":
        return "win"
    if raw == "LOSS":
        return "loss"
    return "breakeven"


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
        decision_id = decision_id_from_execution(ex)
        if decision_id and record_outcome_from_execution(db, decision_id, ex):
            recorded += 1
    return recorded


def recent_closed_trades_feed(
    db,
    user_id: int,
    *,
    limit: int = 25,
    ctrader_account_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Closed gemini gold trades for portal — cTrader StrategyExecution rows."""
    from app.strategy_models import StrategyExecution

    rows = (
        gemini_broker_executions_query(
            db,
            user_id=int(user_id),
            ctrader_account_id=ctrader_account_id,
            closed_only=True,
            outcomes=("WIN", "LOSS", "BREAKEVEN", "CANCELLED"),
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(max(1, min(limit, 50)))
        .all()
    )
    outcome_by_decision: Dict[int, Any] = {}
    if rows:
        decision_ids = [did for ex in rows if (did := decision_id_from_execution(ex))]
        if decision_ids:
            from app.gemini_gold_trader.models import GeminiGoldOutcome

            for o in (
                db.query(GeminiGoldOutcome)
                .filter(GeminiGoldOutcome.decision_id.in_(decision_ids))
                .all()
            ):
                outcome_by_decision[int(o.decision_id)] = o

    out: List[Dict[str, Any]] = []
    for ex in rows:
        decision_id = decision_id_from_execution(ex)
        setup_type = None
        if decision_id and decision_id in outcome_by_decision:
            setup_type = outcome_by_decision[decision_id].setup_type
        hold_min = None
        if ex.fired_at and ex.closed_at:
            hold_min = round((ex.closed_at - ex.fired_at).total_seconds() / 60.0, 1)
        out.append(
            {
                "execution_id": ex.id,
                "decision_id": decision_id,
                "setup_type": setup_type,
                "fired_at": ex.fired_at.isoformat() if ex.fired_at else None,
                "closed_at": ex.closed_at.isoformat() if ex.closed_at else None,
                "direction": ex.direction,
                "outcome": ex.outcome,
                "entry_price": float(ex.entry_price) if ex.entry_price else None,
                "exit_price": float(ex.exit_price) if ex.exit_price else None,
                "pnl_pct": float(ex.pnl_pct) if ex.pnl_pct is not None else None,
                "pnl_usd": float(ex.pnl_usd) if ex.pnl_usd is not None else None,
                "hold_min": hold_min,
                "broker_position_id": ex.ctrader_position_id,
            }
        )
    return out
