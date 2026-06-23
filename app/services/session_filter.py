"""UTC session windows for forex entry filtering."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

from app.services.forex_sessions import (
    ICT_KILLZONE_WINDOWS,
    LIVE_FOREX_SESSIONS,
    LIVE_FOREX_SESSION_ALIASES,
    OTHER_SESSION_WINDOWS,
    in_window,
    overlap_window,
    resolve_live_forex_session_key,
)

# Re-export for callers that imported _in_window from here.
_in_window = in_window


def _build_sessions_dict() -> Dict[str, Tuple[int, int, int, int]]:
    sessions: Dict[str, Tuple[int, int, int, int]] = {}
    sessions.update(LIVE_FOREX_SESSIONS)
    for alias, key in LIVE_FOREX_SESSION_ALIASES.items():
        if alias not in sessions and key in LIVE_FOREX_SESSIONS:
            sessions[alias] = LIVE_FOREX_SESSIONS[key]
    sessions.update(ICT_KILLZONE_WINDOWS)
    sessions.update(OTHER_SESSION_WINDOWS)
    sessions["overlap"] = overlap_window()
    return sessions


SESSIONS: Dict[str, Tuple[int, int, int, int]] = _build_sessions_dict()


def is_in_allowed_session(cfg: dict, now_utc: datetime) -> Tuple[bool, str]:
    """Return (allowed, block_reason_label)."""
    if not cfg.get("sessions_enabled"):
        return True, ""
    allowed = [str(s).lower() for s in (cfg.get("allowed_sessions") or [])]
    custom = cfg.get("session_custom") or {}
    if custom.get("start") and custom.get("end"):
        try:
            sh, sm = [int(x) for x in str(custom["start"]).split(":")[:2]]
            eh, em = [int(x) for x in str(custom["end"]).split(":")[:2]]
            if _in_window(now_utc, sh, sm, eh, em):
                return True, ""
        except Exception:
            pass
    for sid in allowed:
        norm = sid.replace("new_york", "newyork").replace("-", "_")
        key = resolve_live_forex_session_key(norm)
        if key:
            win = LIVE_FOREX_SESSIONS[key]
            if _in_window(now_utc, win[0], win[1], win[2], win[3]):
                return True, ""
            continue
        win = SESSIONS.get(norm)
        if not win:
            continue
        if _in_window(now_utc, win[0], win[1], win[2], win[3]):
            return True, ""
    return False, "session_filter"
