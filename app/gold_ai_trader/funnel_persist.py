"""Persist funnel events to PostgreSQL."""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PERSIST_EVENTS = frozenset({
    "data_blocked",
    "gate_skipped",
    "dedupe_skipped",
    "validator_rejected",
    "news_blocked",
    "claude_take",
    "executed",
    "pending_entry",
    "no_ta_match",
})


def persist_funnel_event(
    db,
    *,
    session: Optional[str],
    event: str,
    setup: Optional[str] = None,
    reason: Optional[str] = None,
    decision_id: Optional[int] = None,
) -> None:
    if event not in _PERSIST_EVENTS:
        return
    try:
        from app.gold_ai_trader.models import GoldAiFunnelEvent

        row = GoldAiFunnelEvent(
            session=session,
            event=event[:32],
            setup_type=(setup or "")[:64] or None,
            reason=(reason or "")[:256] or None,
            decision_id=decision_id,
        )
        db.add(row)
        db.commit()
    except Exception as exc:
        logger.debug("[gold-ai] funnel persist: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass


def recent_funnel_events(db, *, limit: int = 50) -> list:
    from app.gold_ai_trader.models import GoldAiFunnelEvent

    rows = (
        db.query(GoldAiFunnelEvent)
        .order_by(GoldAiFunnelEvent.ts.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "ts": r.ts.isoformat() if r.ts else None,
            "session": r.session,
            "event": r.event,
            "setup_type": r.setup_type,
            "reason": r.reason,
            "decision_id": r.decision_id,
        }
        for r in rows
    ]
