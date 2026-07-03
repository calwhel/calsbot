"""Broker close reconciliation helpers for Gemini Gold Trader."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig
from app.gemini_gold_trader.guardrails import active_ctrader_account_id

logger = logging.getLogger(__name__)


def _gemini_open_query(db, user_id: int, demo_ctid: Optional[str] = None):
    from app.strategy_models import StrategyExecution
    from sqlalchemy import or_

    q = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.user_id == user_id,
            StrategyExecution.symbol == "XAUUSD",
            StrategyExecution.outcome == "OPEN",
            StrategyExecution.notes.like("%gemini_gold_trader%"),
        )
    )
    if demo_ctid:
        ctid = str(demo_ctid).strip()
        q = q.filter(
            or_(
                StrategyExecution.ctrader_account_id == ctid,
                StrategyExecution.ctrader_account_id.is_(None),
                StrategyExecution.ctrader_account_id == "",
            )
        )
    return q.order_by(StrategyExecution.fired_at.desc())


def list_open_executions(
    db,
    user_id: int,
    demo_ctid: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """OPEN gemini_gold StrategyExecution rows (used for cap diagnostics)."""
    rows = _gemini_open_query(db, user_id, demo_ctid=demo_ctid).all()
    out: List[Dict[str, Any]] = []
    for ex in rows:
        out.append(_execution_snapshot(ex))
    return out


def _execution_snapshot(ex) -> Dict[str, Any]:
    from app.services.strategy_executor import (
        _ctrader_order_id_from_execution,
        _ctrader_position_id_from_execution,
    )

    return {
        "execution_id": ex.id,
        "fired_at": ex.fired_at.isoformat() if ex.fired_at else None,
        "direction": ex.direction,
        "entry_price": float(ex.entry_price) if ex.entry_price else None,
        "ctrader_account_id": ex.ctrader_account_id,
        "ctrader_order_id": ex.ctrader_order_id,
        "ctrader_position_id": ex.ctrader_position_id,
        "broker_position_id": _ctrader_position_id_from_execution(ex),
        "broker_order_id": _ctrader_order_id_from_execution(ex),
        "notes": (ex.notes or "")[:240],
    }


def _cancel_orphan_execution(db, ex, *, reason: str) -> Dict[str, Any]:
    """Mark a phantom OPEN row closed so it no longer counts toward caps."""
    snap = _execution_snapshot(ex)
    now = datetime.utcnow()
    suffix = f" | gemini orphan reconcile: {reason}"
    ex.outcome = "CANCELLED"
    ex.closed_at = now
    ex.exit_price = float(ex.entry_price) if ex.entry_price else None
    ex.notes = ((ex.notes or "") + suffix)[:2000]
    db.commit()
    logger.warning(
        "[gemini-gold] cancelled orphan OPEN exec=%s pos=%s order=%s reason=%s",
        ex.id,
        snap.get("broker_position_id"),
        snap.get("broker_order_id"),
        reason,
    )
    return {**snap, "cancel_reason": reason}


def reconcile_orphan_open_executions_sync(
    db,
    *,
    user_id: int,
    demo_ctid: Optional[str],
    broker_open_position_ids: Optional[Set[int]],
    dry_run: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Close OPEN gemini_gold rows with no live broker position on the demo account.

    Returns (rows_before, rows_closed). When ``dry_run`` is True, lists would-be
    closures without mutating the database.
    """
    from app.services.strategy_executor import _ctrader_position_id_from_execution

    rows = _gemini_open_query(db, user_id).all()
    before = [_execution_snapshot(ex) for ex in rows]
    closed: List[Dict[str, Any]] = []

    open_ids: Optional[Set[int]]
    if broker_open_position_ids is not None:
        open_ids = {int(x) for x in broker_open_position_ids}
    else:
        open_ids = None

    ctid_str = str(demo_ctid or "").strip()

    # Broker account is flat — every gemini OPEN row is a phantom cap blocker.
    if open_ids is not None and len(open_ids) == 0 and rows:
        for ex in rows:
            if dry_run:
                closed.append(
                    {
                        **_execution_snapshot(ex),
                        "cancel_reason": f"broker flat on demo ctid {ctid_str or '?'}",
                    }
                )
                continue
            closed.append(
                _cancel_orphan_execution(
                    db,
                    ex,
                    reason=f"broker flat on demo ctid {ctid_str or '?'}",
                )
            )
        return before, closed

    for ex in rows:
        pos_id = _ctrader_position_id_from_execution(ex)

        if pos_id is None:
            reason = "no broker position_id (order never opened)"
        elif open_ids is None:
            continue
        elif pos_id in open_ids:
            continue
        else:
            reason = f"broker position {pos_id} absent on ctid {ctid_str or '?'}"

        if dry_run:
            closed.append({**_execution_snapshot(ex), "cancel_reason": reason})
            continue

        closed.append(_cancel_orphan_execution(db, ex, reason=reason))

    return before, closed


async def reconcile_orphan_open_executions(
    db,
    *,
    cfg: GeminiGoldRuntimeConfig,
    user_id: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Poll demo account positions and cancel gemini OPEN rows with no broker match."""
    from app.models import User
    from app.services.ctrader_client import get_open_position_ids_for_user_with_retry

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        return {"ok": False, "error": "user_not_found", "before": [], "closed": []}

    ctid = active_ctrader_account_id(cfg)
    open_ids = await get_open_position_ids_for_user_with_retry(
        user,
        ctid=ctid,
        attempts=3,
    )

    before, closed = reconcile_orphan_open_executions_sync(
        db,
        user_id=user_id,
        demo_ctid=ctid,
        broker_open_position_ids=open_ids,
        dry_run=dry_run,
    )
    if open_ids is None:
        logger.warning(
            "[gemini-gold] broker open-position poll failed uid=%s ctid=%s "
            "(closed position_id-less orphans only)",
            user_id,
            ctid,
        )
    return {
        "ok": True,
        "broker_open_positions": len(open_ids) if open_ids is not None else None,
        "open_before": len(before),
        "open_after": len(before) - len(closed),
        "orphans_before": before,
        "orphans_closed": closed,
        "dry_run": dry_run,
    }
