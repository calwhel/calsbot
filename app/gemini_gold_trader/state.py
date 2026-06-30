"""In-process runtime status for logs/heartbeat."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class RuntimeStatus:
    status: str = "dormant"
    active_session: Optional[str] = None
    last_scan_at: Optional[str] = None
    last_decision: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    loop_iterations: int = 0


_STATUS = RuntimeStatus()


def get_status() -> Dict[str, Any]:
    return asdict(_STATUS)


def note_scan(session: Optional[str]) -> None:
    _STATUS.status = "scanning" if session else "dormant"
    _STATUS.active_session = session
    _STATUS.last_scan_at = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    _STATUS.loop_iterations += 1


def note_dormant(reason: str = "outside_session") -> None:
    _STATUS.status = "dormant" if reason == "outside_session" else reason
    _STATUS.active_session = None


def note_decision(decision: Dict[str, Any]) -> None:
    _STATUS.last_decision = decision


def note_error(msg: str) -> None:
    _STATUS.last_error = msg[:500]
