"""In-process runtime status for the UI/API."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RuntimeStatus:
    status: str = "dormant"  # dormant | scanning | disabled | killed
    active_session: Optional[str] = None
    last_scan_at: Optional[str] = None
    last_loop_at: Optional[str] = None
    dormant_reason: Optional[str] = None
    last_candidate: Optional[Dict[str, Any]] = None
    last_decision: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    loop_iterations: int = 0
    funnel: Optional[Dict[str, Any]] = None


_STATUS = RuntimeStatus()


def get_status() -> Dict[str, Any]:
    return asdict(_STATUS)


def set_status(**kwargs) -> None:
    for k, v in kwargs.items():
        if hasattr(_STATUS, k):
            setattr(_STATUS, k, v)


def note_scan(session: Optional[str]) -> None:
    _STATUS.status = "scanning" if session else "dormant"
    _STATUS.active_session = session
    _STATUS.dormant_reason = None
    now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    # Emit explicit UTC timestamp for client-side heartbeat age checks.
    _STATUS.last_scan_at = now
    _STATUS.last_loop_at = now
    _STATUS.loop_iterations += 1


def note_loop_tick(*, dormant_reason: Optional[str] = None) -> None:
    """Mark the background loop alive — used every cycle, including outside session."""
    _STATUS.last_loop_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    _STATUS.loop_iterations += 1
    if dormant_reason:
        _STATUS.dormant_reason = dormant_reason
        _STATUS.status = "dormant"
        _STATUS.active_session = None


def note_dormant(reason: str = "outside_session") -> None:
    _STATUS.status = "dormant" if reason == "outside_session" else reason
    _STATUS.active_session = None
    _STATUS.dormant_reason = reason


def note_candidate(candidate: Dict[str, Any]) -> None:
    _STATUS.last_candidate = candidate


def note_decision(decision: Dict[str, Any]) -> None:
    _STATUS.last_decision = decision


def note_error(msg: str) -> None:
    _STATUS.last_error = msg[:500]


def set_funnel(data: Dict[str, Any]) -> None:
    _STATUS.funnel = data
