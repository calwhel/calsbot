"""Scan → candidate → Claude → execute funnel telemetry (in-process, resets daily UTC)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class FunnelCounters:
    day: str = ""
    scans: int = 0
    data_blocked: int = 0
    no_ta_match: int = 0
    ta_detected: int = 0
    htf_skipped: int = 0
    session_skipped: int = 0
    gate_skipped: int = 0
    readiness_skipped: int = 0
    candidates_passed: int = 0
    dedupe_skipped: int = 0
    override_confluence_skipped: int = 0
    claude_called: int = 0
    claude_take: int = 0
    claude_skip: int = 0
    executed: int = 0
    pending_entry: int = 0
    validator_rejected: int = 0
    news_blocked: int = 0
    last_gate_reason: str = ""
    last_dedupe_reason: str = ""
    last_data_block: str = ""
    ta_by_setup: Dict[str, int] = field(default_factory=dict)
    gate_reasons: Dict[str, int] = field(default_factory=dict)


_COUNTERS = FunnelCounters()


def _today() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _maybe_roll_day() -> None:
    today = _today()
    if _COUNTERS.day != today:
        _COUNTERS.day = today
        _COUNTERS.scans = 0
        _COUNTERS.data_blocked = 0
        _COUNTERS.no_ta_match = 0
        _COUNTERS.ta_detected = 0
        _COUNTERS.htf_skipped = 0
        _COUNTERS.session_skipped = 0
        _COUNTERS.gate_skipped = 0
        _COUNTERS.readiness_skipped = 0
        _COUNTERS.candidates_passed = 0
        _COUNTERS.dedupe_skipped = 0
        _COUNTERS.override_confluence_skipped = 0
        _COUNTERS.claude_called = 0
        _COUNTERS.claude_take = 0
        _COUNTERS.claude_skip = 0
        _COUNTERS.executed = 0
        _COUNTERS.pending_entry = 0
        _COUNTERS.validator_rejected = 0
        _COUNTERS.news_blocked = 0
        _COUNTERS.last_gate_reason = ""
        _COUNTERS.last_dedupe_reason = ""
        _COUNTERS.last_data_block = ""
        _COUNTERS.ta_by_setup = {}
        _COUNTERS.gate_reasons = {}


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
    elif event == "ta_detected":
        c.ta_detected += 1
        if setup:
            c.ta_by_setup[setup] = c.ta_by_setup.get(setup, 0) + 1
    elif event == "htf_skipped":
        c.htf_skipped += 1
    elif event == "session_skipped":
        c.session_skipped += 1
    elif event == "gate_skipped":
        c.gate_skipped += 1
        if reason:
            c.last_gate_reason = reason[:120]
            key = reason.split("(")[0].strip()[:40]
            c.gate_reasons[key] = c.gate_reasons.get(key, 0) + 1
    elif event == "candidate_passed":
        c.candidates_passed += 1
    elif event == "readiness_skipped":
        c.readiness_skipped += 1
        if reason:
            key = reason.split("(")[0].strip()[:40]
            c.gate_reasons[key] = c.gate_reasons.get(key, 0) + 1
    elif event == "no_ta_match":
        c.no_ta_match += 1
    elif event == "dedupe_skipped":
        c.dedupe_skipped += 1
        if reason:
            c.last_dedupe_reason = reason[:120]
    elif event == "override_confluence_skipped":
        c.override_confluence_skipped += 1
        if reason:
            c.last_gate_reason = reason[:120]
            key = reason.split("(")[0].strip()[:40]
            c.gate_reasons[key] = c.gate_reasons.get(key, 0) + 1
    elif event == "claude_called":
        c.claude_called += 1
    elif event == "claude_take":
        c.claude_take += 1
    elif event == "claude_skip":
        c.claude_skip += 1
    elif event == "executed":
        c.executed += 1
    elif event == "pending_entry":
        c.pending_entry += 1
    elif event == "validator_rejected":
        c.validator_rejected += 1
    elif event == "news_blocked":
        c.news_blocked += 1


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
    from app.gold_ai_trader.funnel_persist import persist_funnel_event

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
        "no_ta_match": c.no_ta_match,
        "ta_detected": c.ta_detected,
        "htf_skipped": c.htf_skipped,
        "session_skipped": c.session_skipped,
        "gate_skipped": c.gate_skipped,
        "readiness_skipped": c.readiness_skipped,
        "candidates_passed": c.candidates_passed,
        "dedupe_skipped": c.dedupe_skipped,
        "override_confluence_skipped": c.override_confluence_skipped,
        "claude_called": c.claude_called,
        "claude_take": c.claude_take,
        "claude_skip": c.claude_skip,
        "executed": c.executed,
        "pending_entry": c.pending_entry,
        "validator_rejected": c.validator_rejected,
        "news_blocked": c.news_blocked,
        "last_gate_reason": c.last_gate_reason,
        "last_dedupe_reason": c.last_dedupe_reason,
        "last_data_block": c.last_data_block,
        "ta_by_setup": dict(c.ta_by_setup),
        "gate_reasons": dict(c.gate_reasons),
    }
