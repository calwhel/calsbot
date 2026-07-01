"""Gold-module entry pending / limit placement (demo only)."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Tuple

from app.gold_ai_trader.db_thread import db_commit, run_in_db_thread

logger = logging.getLogger(__name__)


def broker_limit_supported() -> bool:
    """True only if shared ctrader client exposes a limit/pending helper."""
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
) -> Tuple[Optional[dict], Optional[str]]:
    """Attempt broker LIMIT via shared client if available. Returns (result, error)."""
    if not broker_limit_supported():
        return None, "broker LIMIT helper not available in ctrader_client"

    from app.services import ctrader_client as cc
    from app.gold_ai_trader.config import SYMBOL
    from app.gold_ai_trader.executor import _parse_prices

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
            volume_lots=max(0.01, float(cfg.demo_lot_size or cfg.min_lot or 0.01)),
            limit_price=entry,
            entry_price=entry,
            stop_loss_price=sl,
            take_profit_price=tp,
            label="GoldAITrader",
            execution_id=decision_id,
        )
        return result, None
    except TypeError as e:
        return None, f"limit helper signature mismatch: {e}"
    except Exception as e:
        return None, str(e)


def session_end_utc(now: datetime, session: str, cfg) -> datetime:
    day = now.replace(hour=0, minute=0, second=0, microsecond=0)
    if session == "london":
        return day.replace(hour=int(cfg.london_end_hour), minute=0, second=0, microsecond=0)
    if session == "new_york":
        return day.replace(hour=int(cfg.ny_end_hour), minute=0, second=0, microsecond=0)
    if session == "asia":
        from app.services.forex_sessions import LIVE_FOREX_SESSIONS

        _, _, end_h, end_m = LIVE_FOREX_SESSIONS["asia"]
        return day.replace(hour=int(end_h), minute=int(end_m), second=0, microsecond=0)
    return now + timedelta(hours=1)


def compute_pending_expiry(now: datetime, session: str, cfg, timeout_min: int) -> datetime:
    by_timeout = now + timedelta(minutes=max(1, int(timeout_min)))
    by_session = session_end_utc(now, session, cfg)
    return min(by_timeout, by_session) if by_session > now else by_timeout


def entry_price_touched(direction: str, spot: float, entry: float, tolerance: float) -> bool:
    """Simulate limit fill when spot reaches entry (within tolerance)."""
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
    """Store gold-module entry watch pending (used when broker LIMIT unavailable)."""
    from app.gold_ai_trader.executor import _parse_prices
    from app.gold_ai_trader.models import GoldAiPendingOrder

    parsed = _parse_prices(decision)
    if not parsed:
        return None
    direction, entry, sl, tp = parsed
    now = datetime.utcnow()
    expires = compute_pending_expiry(
        now, session, cfg, getattr(cfg, "pending_entry_timeout_min", 30)
    )
    row = GoldAiPendingOrder(
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
        notes="broker LIMIT unavailable; gold entry-watch pending",
    )
    db.add(row)
    await db_commit(db)
    await run_in_db_thread(db.refresh, row)
    logger.info(
        "[gold-ai-trader] entry-watch pending id=%s decision=%s entry=%s expires=%s",
        row.id,
        decision_id,
        entry,
        expires.isoformat(),
    )
    return row.id


async def sync_pending_entries(db, cfg, spot: float) -> int:
    """Expire or fill gold pending entry watches. Returns fills this pass."""
    from app.gold_ai_trader.models import GoldAiPendingOrder
    from app.gold_ai_trader.executor import execute_take_market

    def _load_pending():
        now = datetime.utcnow()
        pending = (
            db.query(GoldAiPendingOrder)
            .filter(GoldAiPendingOrder.status == "pending")
            .order_by(GoldAiPendingOrder.created_at.asc())
            .all()
        )
        return now, pending

    now, pending = await run_in_db_thread(_load_pending)
    filled = 0
    for row in pending:
        if row.method == "broker_limit":
            continue
        if row.expires_at and now >= row.expires_at:
            row.status = "expired"
            row.notes = (row.notes or "") + " | expired"
            await db_commit(db)
            logger.info("[gold-ai-trader] pending expired id=%s decision=%s", row.id, row.decision_id)
            continue

        from app.gold_ai_trader.models import GoldAiDecision

        dec_row = await run_in_db_thread(
            lambda: db.query(GoldAiDecision).filter(GoldAiDecision.id == row.decision_id).first()
        )
        if not dec_row or not dec_row.decision:
            row.status = "cancelled"
            await db_commit(db)
            continue

        # tolerance ~0.05 ATR not available here — use 0.15 absolute gold dollars min
        tol = max(0.15, abs(row.entry_price) * 0.00005)
        if not entry_price_touched(row.direction, spot, float(row.entry_price), tol):
            continue

        decision = dict(dec_row.decision)
        setup_type = str(getattr(dec_row, "candidate_type", "") or "")
        candidate_direction = str(
            decision.get("direction") or row.direction or ""
        )
        timing_ctx = {
            "decision_ts": dec_row.ts.isoformat() + "Z" if getattr(dec_row, "ts", None) else None,
            "validated_ts": None,
            "enqueued_ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "broker_ack_ts": None,
        }
        exec_id = await execute_take_market(
            db=db,
            cfg=cfg,
            decision=decision,
            decision_id=row.decision_id,
            entry_note=f"entry-watch fill @ spot {spot:.2f} (pending #{row.id})",
            timing_ctx=timing_ctx,
            fire_context={
                "setup_type": setup_type,
                "candidate_direction": candidate_direction,
                "setup_detail": "",
                "user_id": cfg.demo_user_id,
            },
        )
        if exec_id is None and timing_ctx.get("block_reason"):
            row.status = "cancelled"
            row.notes = (
                (row.notes or "")
                + f" | fire_time_blocked {timing_ctx.get('block_reason')}"
            )
            await db_commit(db)
            logger.warning(
                "[gold-ai-trader] pending fire_time blocked id=%s decision=%s reason=%s",
                row.id,
                row.decision_id,
                timing_ctx.get("block_reason"),
            )
            continue
        if exec_id:
            row.status = "filled"
            row.fill_execution_id = exec_id
            row.notes = (row.notes or "") + f" | filled exec #{exec_id}"
            dec_row.executed = True
            dec_row.execution_id = exec_id
            await db_commit(db)
            filled += 1
            logger.info(
                "[gold-ai-trader] entry-watch filled pending=%s exec=%s", row.id, exec_id
            )
            logger.info(
                "[gold-ai-latency] decision_id=%s setup=pending_entry_watch decision_ts=%s "
                "validated_ts=%s enqueued_ts=%s broker_ack_ts=%s exec_id=%s block_reason=",
                row.decision_id,
                timing_ctx.get("decision_ts"),
                timing_ctx.get("validated_ts"),
                timing_ctx.get("enqueued_ts"),
                timing_ctx.get("broker_ack_ts"),
                exec_id,
            )
    return filled


def pending_status_label(db, pending_id: int) -> str:
    """Human-readable Telegram status for a pending entry row."""
    from app.gold_ai_trader.models import GoldAiPendingOrder

    row = db.query(GoldAiPendingOrder).filter(GoldAiPendingOrder.id == pending_id).first()
    if not row:
        return f"entry pending #{pending_id}"
    if row.method == "broker_limit" and row.broker_order_id:
        return (
            f"broker LIMIT placed (#{pending_id}, cTrader order {row.broker_order_id}) "
            f"@ {float(row.entry_price):.2f}"
        )
    return (
        f"entry-watch #{pending_id} @ {float(row.entry_price):.2f} "
        "(software watch — no broker limit; fills with market when price touches)"
    )


def pending_entry_count(db, user_id: int) -> int:
    from app.gold_ai_trader.models import GoldAiPendingOrder

    return (
        db.query(GoldAiPendingOrder)
        .filter(GoldAiPendingOrder.status == "pending")
        .count()
    )
