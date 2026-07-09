"""Persist Gemini Gold funnel events to PostgreSQL."""
from __future__ import annotations

import logging
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)

_PERSIST_EVENTS = frozenset({
    "scan",
    "data_blocked",
    "chart_failed",
    "gemini_called",
    "gemini_take",
    "gemini_skip",
    "validator_rejected",
    "stale_entry_blocked",
    "execution_blocked",
    "executed",
    "pending_entry",
    "orb_detected",
    "orb_executed",
})
_SCAN_PERSIST_INTERVAL_S = max(
    5.0,
    float(os.environ.get("GEMINI_GOLD_SCAN_HEARTBEAT_PERSIST_S", "20")),
)
_last_scan_persist_mono = 0.0


def persist_funnel_event(
    db,
    *,
    session: Optional[str],
    event: str,
    setup: Optional[str] = None,
    reason: Optional[str] = None,
    decision_id: Optional[int] = None,
) -> None:
    global _last_scan_persist_mono
    if event not in _PERSIST_EVENTS:
        return
    if event == "scan":
        now_m = time.monotonic()
        if (now_m - _last_scan_persist_mono) < _SCAN_PERSIST_INTERVAL_S:
            return
        _last_scan_persist_mono = now_m
    try:
        from app.gemini_gold_trader.models import GeminiGoldFunnelEvent

        row = GeminiGoldFunnelEvent(
            session=session,
            event=event[:32],
            setup_type=(setup or "")[:64] or None,
            reason=(reason or "")[:256] or None,
            decision_id=decision_id,
        )
        db.add(row)
        db.commit()
    except Exception as exc:
        logger.debug("[gemini-gold] funnel persist: %s", exc)
        try:
            db.rollback()
        except Exception:
            pass


def recent_funnel_events(db, *, limit: int = 50) -> list:
    from app.gemini_gold_trader.models import GeminiGoldFunnelEvent

    rows = (
        db.query(GeminiGoldFunnelEvent)
        .order_by(GeminiGoldFunnelEvent.ts.desc())
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
