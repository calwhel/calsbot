"""
Single source of truth for live forex scan/fire session windows (fixed UTC).

London 06:00–09:00 | New York 12:00–15:00 | Asia 01:00–04:00
(Asia = 02:00–05:00 UK during BST; stored as UTC — no Oct DST drift.)

Used by: live-forex scan gate, wizard session_filter, Claude confirm gate,
and legacy filters.session in strategy_executor (live session ids only).

ICT killzones (asian_kz 20–23, etc.) are a separate concept — see ICT_KILLZONE_WINDOWS.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional, Tuple

# (start_h, start_m, end_h, end_m) — end exclusive, same as session_filter._in_window
SessionWindow = Tuple[int, int, int, int]

LIVE_FOREX_SESSIONS: Dict[str, SessionWindow] = {
    "london": (6, 0, 9, 0),
    "new_york": (12, 0, 15, 0),
    "asia": (1, 0, 4, 0),
}

# Wizard / legacy id → canonical live-forex session key
LIVE_FOREX_SESSION_ALIASES: Dict[str, str] = {
    "london": "london",
    "europe": "london",
    "newyork": "new_york",
    "new_york": "new_york",
    "ny": "new_york",
    "asia": "asia",
    "asian": "asia",
}

# ICT entry-condition killzones — NOT live-forex scan/fire windows (strategy_ta.py).
ICT_KILLZONE_WINDOWS: Dict[str, SessionWindow] = {
    "london_kz": (7, 0, 9, 0),
    "ny_kz": (12, 0, 14, 0),
    "asian_kz": (20, 0, 23, 0),
}

# Optional sessions outside live-forex scan/fire (paper / discovery only).
OTHER_SESSION_WINDOWS: Dict[str, SessionWindow] = {
    "sydney": (21, 0, 6, 0),
    "tokyo": (0, 0, 9, 0),
}

_ICT_ANY_KZ = frozenset(ICT_KILLZONE_WINDOWS.keys())


def in_window(now: datetime, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    t = now.hour * 60 + now.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start > end:
        return t >= start or t < end
    return start <= t < end


def _normalize_session_id(session_id: str) -> str:
    return str(session_id or "").lower().strip().replace("-", "_")


def resolve_live_forex_session_key(session_id: str) -> Optional[str]:
    sid = _normalize_session_id(session_id)
    if sid in LIVE_FOREX_SESSIONS:
        return sid
    return LIVE_FOREX_SESSION_ALIASES.get(sid)


def active_live_forex_session(now_utc: datetime) -> Optional[str]:
    for sid, win in LIVE_FOREX_SESSIONS.items():
        if in_window(now_utc, win[0], win[1], win[2], win[3]):
            return sid
    return None


def in_live_forex_session(now_utc: Optional[datetime] = None) -> bool:
    return active_live_forex_session(now_utc or datetime.utcnow()) is not None


def is_named_session_active(session_id: str, now_utc: datetime) -> bool:
    """True if `session_id` is active at `now_utc` (live forex, ICT KZ, or other)."""
    sid = _normalize_session_id(session_id)
    if sid == "any_kz":
        return any(
            is_named_session_active(k, now_utc) for k in _ICT_ANY_KZ
        )
    key = resolve_live_forex_session_key(sid)
    if key:
        win = LIVE_FOREX_SESSIONS[key]
        return in_window(now_utc, win[0], win[1], win[2], win[3])
    if sid in ICT_KILLZONE_WINDOWS:
        win = ICT_KILLZONE_WINDOWS[sid]
        return in_window(now_utc, win[0], win[1], win[2], win[3])
    if sid in OTHER_SESSION_WINDOWS:
        win = OTHER_SESSION_WINDOWS[sid]
        return in_window(now_utc, win[0], win[1], win[2], win[3])
    return False


def live_forex_session_allowed(
    cfg: dict,
    now_utc: datetime,
) -> Tuple[bool, str]:
    """
    Gate for live forex scan + fire.

    Always requires one of the three canonical windows. When the wizard enables
    session restriction, further narrows to the selected session(s).
    """
    if not in_live_forex_session(now_utc):
        return False, "outside_live_forex_session"
    if cfg.get("sessions_enabled"):
        from app.services.session_filter import is_in_allowed_session

        ok, reason = is_in_allowed_session(cfg, now_utc)
        if not ok:
            return ok, reason or "session_filter"
    return True, ""


def session_filter_matches_confirm_gate(now_utc: datetime) -> bool:
    """True when session_filter (any live session selected) and confirm gate agree."""
    return in_live_forex_session(now_utc)


def legacy_hour_bucket(session_id: str) -> Optional[Tuple[int, int]]:
    """
    Hour-only (start, end) buckets for code paths that still use hour <= h < end.
    Prefer in_window / is_named_session_active for new code.
    """
    sid = _normalize_session_id(session_id)
    key = resolve_live_forex_session_key(sid)
    if key:
        sh, sm, eh, em = LIVE_FOREX_SESSIONS[key]
        return (sh, eh)
    if sid in ICT_KILLZONE_WINDOWS:
        sh, sm, eh, em = ICT_KILLZONE_WINDOWS[sid]
        return (sh, eh)
    if sid in OTHER_SESSION_WINDOWS:
        sh, sm, eh, em = OTHER_SESSION_WINDOWS[sid]
        return (sh, eh)
    return None


def build_executor_session_hours() -> Dict[str, Tuple[int, int]]:
    """Legacy filters.session table — live ids from this module; ICT KZ unchanged."""
    out: Dict[str, Tuple[int, int]] = {}
    for sid, win in LIVE_FOREX_SESSIONS.items():
        out[sid] = (win[0], win[2])
    for alias, key in LIVE_FOREX_SESSION_ALIASES.items():
        win = LIVE_FOREX_SESSIONS[key]
        out[alias] = (win[0], win[2])
    for sid, win in ICT_KILLZONE_WINDOWS.items():
        out[sid] = (win[0], win[2])
    for sid, win in OTHER_SESSION_WINDOWS.items():
        out[sid] = (win[0], win[2])
    out["overlap"] = out.get("new_york", (12, 15))
    return out
