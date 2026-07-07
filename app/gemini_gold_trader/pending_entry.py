"""Gemini Gold limit / entry-watch pending orders."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from app.gemini_gold_trader.db_thread import db_commit, run_in_db_thread

logger = logging.getLogger(__name__)


def broker_limit_supported() -> bool:
    try:
        from app.services import ctrader_client as cc

        for name in (
            "place_limit_order_resilient",
            "place_limit_order",
            "place_pending_order",
            "place_pending_order_resilient",
        ):
            if callable(getattr(cc, name, None)):
                return True
    except Exception:
        pass
    return False


async def try_place_broker_limit(
    *,
    user,
    prefs,
    ctid: int,
    cfg,
    decision: Dict[str, Any],
    decision_id: int,
    volume_lots: float,
) -> Tuple[Optional[dict], Optional[str]]:
    if not broker_limit_supported():
        return None, "broker LIMIT helper not available"

    from app.services import ctrader_client as cc
    from app.gemini_gold_trader.config import SYMBOL
    from app.gemini_gold_trader.executor import _parse_prices

    parsed = _parse_prices(decision)
    if not parsed:
        return None, "invalid entry/sl/tp"
    direction, entry, sl, tp = parsed

    fn = None
    for name in (
        "place_limit_order_resilient",
        "place_limit_order",
        "place_pending_order",
        "place_pending_order_resilient",
    ):
        candidate = getattr(cc, name, None)
        if callable(candidate):
            fn = candidate
            break
    if not fn:
        return None, "no limit function"

    try:
        result = await fn(
            user_id=user.id,
            access_token=prefs.ctrader_access_token,
            ctid=ctid,
            prefs=prefs,
            symbol_name=SYMBOL,
            direction=direction,
            volume_lots=max(0.01, float(volume_lots or 0.01)),
            limit_price=entry,
            entry_price=entry,
            stop_loss_price=sl,
            take_profit_price=tp,
            label="GeminiGold",
            execution_id=decision_id,
        )
        return result, None
    except Exception as e:
        return None, str(e)


def compute_pending_expiry(now: datetime, session: str, timeout_min: int) -> datetime:
    return now + timedelta(minutes=max(1, int(timeout_min)))


def entry_price_touched(direction: str, spot: float, entry: float, tolerance: float) -> bool:
    if direction == "LONG":
        return spot <= entry + tolerance
    return spot >= entry - tolerance


async def create_entry_watch_pending(
    db,
    *,
    cfg,
    decision: Dict[str, Any],
    decision_id: int,
    session: str,
) -> Optional[int]:
    from app.gemini_gold_trader.executor import _parse_prices
    from app.gemini_gold_trader.models import GeminiGoldPendingOrder

    parsed = _parse_prices(decision)
    if not parsed:
        return None
    direction, entry, sl, tp = parsed
    now = datetime.utcnow()
    expires = compute_pending_expiry(
        now, session, getattr(cfg, "pending_entry_timeout_min", 30)
    )
    row = GeminiGoldPendingOrder(
        decision_id=decision_id,
        session=session,
        direction=direction,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        status="pending",
        method="entry_watch",
        created_at=now,
        expires_at=expires,
        notes="entry-watch pending",
    )
    db.add(row)
    await db_commit(db)
    await run_in_db_thread(db.refresh, row)
    logger.info(
        "[gemini-gold] entry-watch pending id=%s decision=%s entry=%s",
        row.id,
        decision_id,
        entry,
    )
    return row.id


def pending_status_label(pending_id: int) -> str:
    return f"pending entry watch #{pending_id}"


async def sync_pending_entries(db, cfg, spot: float) -> int:
    """Expire or fill entry-watch pending rows. Returns fills this pass."""
    from app.gemini_gold_trader.executor import execute_take_market
    from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldPendingOrder

    def _load_pending():
        now = datetime.utcnow()
        pending = (
            db.query(GeminiGoldPendingOrder)
            .filter(GeminiGoldPendingOrder.status == "pending")
            .order_by(GeminiGoldPendingOrder.created_at.asc())
            .all()
        )
        return now, pending

    now, pending = await run_in_db_thread(_load_pending)
    filled = 0
    for row in pending:
        if row.expires_at and now >= row.expires_at:
            row.status = "expired"
            row.notes = (row.notes or "") + " | expired"
            await db_commit(db)
            continue

        dec_row = await run_in_db_thread(
            lambda: db.query(GeminiGoldDecision).filter(GeminiGoldDecision.id == row.decision_id).first()
        )
        if not dec_row or not dec_row.decision:
            row.status = "cancelled"
            await db_commit(db)
            continue

        tol = max(0.15, abs(row.entry_price) * 0.00005)
        if not entry_price_touched(row.direction, spot, float(row.entry_price), tol):
            continue

        decision = dict(dec_row.decision)
        exec_id = await execute_take_market(
            db=db,
            cfg=cfg,
            decision=decision,
            decision_id=row.decision_id,
            spot_hint=spot,
        )
        if exec_id and exec_id > 0:
            row.status = "filled"
            row.fill_execution_id = exec_id
            dec_row.executed = True
            dec_row.execution_id = exec_id
            await db_commit(db)
            filled += 1
            try:
                from app.gemini_gold_trader.guardrails import is_live_execution_mode
                from app.gemini_gold_trader.telegram_notify import notify_decision

                await notify_decision(
                    session=str(row.session or getattr(dec_row, "session", None) or ""),
                    decision=decision,
                    action="TAKE",
                    confidence=int(getattr(dec_row, "confidence", None) or 0),
                    executed=True,
                    execution_id=exec_id,
                    dry_run=bool(getattr(cfg, "dry_run", False)),
                    execution_mode="live" if is_live_execution_mode(cfg) else "demo",
                    fill_kind="entry_watch",
                )
            except Exception as exc:
                logger.warning(
                    "[gemini-gold] entry-watch fill notify failed decision=%s: %s",
                    row.decision_id,
                    exc,
                )
            break
    return filled
