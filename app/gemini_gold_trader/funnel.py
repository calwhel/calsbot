"""Scan → Gemini → execute funnel telemetry (in-process, resets daily UTC)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class FunnelCounters:
    day: str = ""
    scans: int = 0
    data_blocked: int = 0
    chart_failed: int = 0
    gemini_called: int = 0
    gemini_take: int = 0
    gemini_skip: int = 0
    validator_rejected: int = 0
    stale_entry_blocked: int = 0
    executed: int = 0
    pending_entry: int = 0
    orb_detected: int = 0
    orb_executed: int = 0
    last_data_block: str = ""
    last_validator_reason: str = ""
    take_by_setup: Dict[str, int] = field(default_factory=dict)


_COUNTERS = FunnelCounters()


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _maybe_roll_day() -> None:
    today = _today()
    if _COUNTERS.day != today:
        _COUNTERS.day = today
        _COUNTERS.scans = 0
        _COUNTERS.data_blocked = 0
        _COUNTERS.chart_failed = 0
        _COUNTERS.gemini_called = 0
        _COUNTERS.gemini_take = 0
        _COUNTERS.gemini_skip = 0
        _COUNTERS.validator_rejected = 0
        _COUNTERS.stale_entry_blocked = 0
        _COUNTERS.executed = 0
        _COUNTERS.pending_entry = 0
        _COUNTERS.orb_detected = 0
        _COUNTERS.orb_executed = 0
        _COUNTERS.last_data_block = ""
        _COUNTERS.last_validator_reason = ""
        _COUNTERS.take_by_setup = {}


def _record_counters(
    event: str,
    *,
    setup: Optional[str] = None,
    reason: Optional[str] = None,
) -> None:
    _maybe_roll_day()
    c = _COUNTERS
    if event == "scan":
        c.scans += 1
    elif event == "data_blocked":
        c.data_blocked += 1
        if reason:
            c.last_data_block = reason[:120]
    elif event == "chart_failed":
        c.chart_failed += 1
    elif event == "gemini_called":
        c.gemini_called += 1
    elif event == "gemini_take":
        c.gemini_take += 1
        if setup:
            c.take_by_setup[setup] = c.take_by_setup.get(setup, 0) + 1
    elif event == "gemini_skip":
        c.gemini_skip += 1
    elif event == "validator_rejected":
        c.validator_rejected += 1
        if reason:
            c.last_validator_reason = reason[:120]
    elif event == "stale_entry_blocked":
        c.stale_entry_blocked += 1
    elif event == "executed":
        c.executed += 1
    elif event == "pending_entry":
        c.pending_entry += 1
    elif event == "orb_detected":
        c.orb_detected += 1
    elif event == "orb_executed":
        c.orb_executed += 1


def record(
    event: str,
    *,
    setup: Optional[str] = None,
    reason: Optional[str] = None,
    db=None,
    session: Optional[str] = None,
    decision_id: Optional[int] = None,
) -> None:
    _record_counters(event, setup=setup, reason=reason)
    if db is None:
        return
    from app.gemini_gold_trader.funnel_persist import persist_funnel_event

    persist_funnel_event(
        db,
        session=session,
        event=event,
        setup=setup,
        reason=reason,
        decision_id=decision_id,
    )


def snapshot() -> Dict[str, Any]:
    _maybe_roll_day()
    c = _COUNTERS
    return {
        "day": c.day,
        "scans": c.scans,
        "data_blocked": c.data_blocked,
        "chart_failed": c.chart_failed,
        "gemini_called": c.gemini_called,
        "gemini_take": c.gemini_take,
        "gemini_skip": c.gemini_skip,
        "validator_rejected": c.validator_rejected,
        "stale_entry_blocked": c.stale_entry_blocked,
        "executed": c.executed,
        "pending_entry": c.pending_entry,
        "orb_detected": c.orb_detected,
        "orb_executed": c.orb_executed,
        "last_data_block": c.last_data_block,
        "last_validator_reason": c.last_validator_reason,
        "take_by_setup": dict(c.take_by_setup),
    }
