"""
Single source of truth for live forex scan/fire session windows (fixed UTC).

London 07:00–16:00 | New York 12:00–21:00 | Asia 01:00–04:00

Used by: live-forex scan gate, wizard session_filter, Claude confirm gate,
forex_engine / eval_forex_session, portal UI, Gold AI Trader, backtest buckets.

ICT killzones (asian_kz 20–23, etc.) are a separate concept — see ICT_KILLZONE_WINDOWS.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# (start_h, start_m, end_h, end_m) — end exclusive
SessionWindow = Tuple[int, int, int, int]

LIVE_FOREX_SESSIONS: Dict[str, SessionWindow] = {
    "london": (7, 0, 16, 0),
    "new_york": (12, 0, 21, 0),
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


def overlap_window() -> SessionWindow:
    """London ∩ New York — deepest liquidity (derived, not stored separately)."""
    lon = LIVE_FOREX_SESSIONS["london"]
    ny = LIVE_FOREX_SESSIONS["new_york"]
    return (
        max(lon[0], ny[0]),
        max(lon[1], ny[1]),
        min(lon[2], ny[2]),
        min(lon[3], ny[3]),
    )


def format_window_compact(win: SessionWindow) -> str:
    """Human label e.g. 7–16 when minutes are zero."""
    sh, sm, eh, em = win
    if sm or em:
        return f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d}"
    return f"{sh}–{eh}"


def format_window_clock(win: SessionWindow) -> str:
    """Human label e.g. 07:00–16:00."""
    sh, sm, eh, em = win
    return f"{sh:02d}:{sm:02d}–{eh:02d}:{em:02d}"


def hour_bucket(session_id: str) -> Optional[Tuple[int, int]]:
    """Start/end UTC hours (end exclusive) for replay/backtest hour checks."""
    sid = _normalize_session_id(session_id)
    if sid == "overlap":
        ov = overlap_window()
        return (ov[0], ov[2])
    key = resolve_live_forex_session_key(sid)
    if key:
        win = LIVE_FOREX_SESSIONS[key]
        return (win[0], win[2])
    if sid in ICT_KILLZONE_WINDOWS:
        win = ICT_KILLZONE_WINDOWS[sid]
        return (win[0], win[2])
    if sid in OTHER_SESSION_WINDOWS:
        win = OTHER_SESSION_WINDOWS[sid]
        return (win[0], win[2])
    return None


def build_hour_buckets() -> Dict[str, Tuple[int, int]]:
    """All named session hour buckets for backtest/scanner (single source)."""
    out: Dict[str, Tuple[int, int]] = {}
    for sid, win in LIVE_FOREX_SESSIONS.items():
        out[sid] = (win[0], win[2])
    for alias, key in LIVE_FOREX_SESSION_ALIASES.items():
        win = LIVE_FOREX_SESSIONS[key]
        out[alias] = (win[0], win[2])
    ov = overlap_window()
    out["overlap"] = (ov[0], ov[2])
    for sid, win in OTHER_SESSION_WINDOWS.items():
        out[sid] = (win[0], win[2])
    for sid, win in ICT_KILLZONE_WINDOWS.items():
        out[sid] = (win[0], win[2])
    out["europe"] = out["london"]
    return out


def wizard_entry_filter_options() -> List[Dict[str, str]]:
    """Wizard entry-filter checkboxes — all windows derived from this module."""
    labels = {
        "sydney": "Sydney",
        "tokyo": "Tokyo",
        "london": "London",
        "newyork": "New York",
        "london_kz": "London KZ",
        "ny_kz": "NY KZ",
    }
    options: List[Dict[str, str]] = []
    for sid, win in OTHER_SESSION_WINDOWS.items():
        wiz_id = "newyork" if sid == "new_york" else sid
        options.append(
            {
                "id": wiz_id,
                "label": labels.get(wiz_id, sid.replace("_", " ").title()),
                "time": format_window_clock(win),
            }
        )
    for key, win in (
        ("london", LIVE_FOREX_SESSIONS["london"]),
        ("newyork", LIVE_FOREX_SESSIONS["new_york"]),
    ):
        options.append(
            {
                "id": key,
                "label": labels[key],
                "time": format_window_clock(win),
            }
        )
    for sid, win in ICT_KILLZONE_WINDOWS.items():
        options.append(
            {
                "id": sid,
                "label": labels.get(sid, sid),
                "time": format_window_clock(win),
            }
        )
    return options


def live_forex_timeline_specs() -> List[Dict[str, Any]]:
    """Live-forex panel session bars — derived from shared windows."""
    colors = {
        "sydney": "#6366f1",
        "asian": "#f59e0b",
        "london": "#3fb68b",
        "new_york": "#e5484d",
    }
    specs: List[Dict[str, Any]] = []
    syd = OTHER_SESSION_WINDOWS["sydney"]
    specs.append(
        {
            "name": "Sydney",
            "open": syd[0],
            "close": syd[2],
            "tz": "UTC",
            "color": colors["sydney"],
        }
    )
    asia = LIVE_FOREX_SESSIONS["asia"]
    specs.append(
        {
            "name": "Asian",
            "open": asia[0],
            "close": asia[2],
            "tz": "UTC",
            "color": colors["asian"],
        }
    )
    lon = LIVE_FOREX_SESSIONS["london"]
    specs.append(
        {
            "name": "London",
            "open": lon[0],
            "close": lon[2],
            "tz": "UTC",
            "color": colors["london"],
        }
    )
    ny = LIVE_FOREX_SESSIONS["new_york"]
    specs.append(
        {
            "name": "New York",
            "open": ny[0],
            "close": ny[2],
            "tz": "UTC",
            "color": colors["new_york"],
        }
    )
    return specs


def portal_session_ui_specs() -> List[Dict[str, str]]:
    """Portal/wizard chip labels — derived from this module only."""
    asia = LIVE_FOREX_SESSIONS["asia"]
    london = LIVE_FOREX_SESSIONS["london"]
    ny = LIVE_FOREX_SESSIONS["new_york"]
    overlap = overlap_window()
    return [
        {
            "id": "asian",
            "cfg_id": "asian",
            "label": "Asian",
            "chip": f"{format_window_compact(asia)} UTC",
            "clock": f"{format_window_clock(asia)} UTC",
        },
        {
            "id": "london",
            "cfg_id": "london",
            "label": "London",
            "chip": f"{format_window_compact(london)} UTC",
            "clock": f"{format_window_clock(london)} UTC",
        },
        {
            "id": "new_york",
            "cfg_id": "new_york",
            "label": "US / New York",
            "chip": f"{format_window_compact(ny)} UTC",
            "clock": f"{format_window_clock(ny)} UTC",
        },
        {
            "id": "overlap",
            "cfg_id": "overlap",
            "label": "London/US Overlap",
            "chip": f"{format_window_compact(overlap)} UTC",
            "clock": f"{format_window_clock(overlap)} UTC",
        },
    ]


def gold_ai_session_hours() -> Dict[str, Dict[str, int]]:
    """London/NY hours for Gold AI Trader UI + runtime (from shared windows)."""
    lon = LIVE_FOREX_SESSIONS["london"]
    ny = LIVE_FOREX_SESSIONS["new_york"]
    return {
        "london": {"start_hour": lon[0], "end_hour": lon[2]},
        "new_york": {"start_hour": ny[0], "end_hour": ny[2]},
    }


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
    """True if `session_id` is active at `now_utc` (live forex, overlap, ICT KZ, or other)."""
    sid = _normalize_session_id(session_id)
    if sid == "any_kz":
        return any(is_named_session_active(k, now_utc) for k in _ICT_ANY_KZ)
    if sid == "overlap":
        ov = overlap_window()
        return in_window(now_utc, ov[0], ov[1], ov[2], ov[3])
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
    """Alias for hour_bucket — legacy executor filters.session table."""
    return hour_bucket(session_id)


def build_executor_session_hours() -> Dict[str, Tuple[int, int]]:
    """Legacy filters.session table — live ids from this module; ICT KZ unchanged."""
    return build_hour_buckets()


def build_forex_engine_session_specs() -> Dict[str, Tuple[str, str, int, int, int, int]]:
    """
    SessionWindow specs for forex_engine.SESSIONS:
    id → (id, label, open_h, open_m, close_h, close_m)
    """
    specs: Dict[str, Tuple[str, str, int, int, int, int]] = {}
    lon = LIVE_FOREX_SESSIONS["london"]
    specs["london"] = ("london", "London", lon[0], lon[1], lon[2], lon[3])
    ny = LIVE_FOREX_SESSIONS["new_york"]
    specs["ny"] = ("ny", "New York", ny[0], ny[1], ny[2], ny[3])
    asia = LIVE_FOREX_SESSIONS["asia"]
    specs["asian"] = ("asian", "Asian", asia[0], asia[1], asia[2], asia[3])
    syd = OTHER_SESSION_WINDOWS["sydney"]
    specs["sydney"] = ("sydney", "Sydney", syd[0], syd[1], syd[2], syd[3])
    ov = overlap_window()
    specs["overlap"] = ("overlap", "London/NY Overlap", ov[0], ov[1], ov[2], ov[3])
    return specs


def build_session_alert_windows() -> List[Tuple[str, str, int, int]]:
    """(session_id, display_label, open_hour_utc, open_min_utc) for session-open alerts."""
    out: List[Tuple[str, str, int, int]] = []
    lon = LIVE_FOREX_SESSIONS["london"]
    out.extend(
        [
            ("london", "London Session", lon[0], lon[1]),
            ("europe", "Europe Session", lon[0], lon[1]),
        ]
    )
    ny = LIVE_FOREX_SESSIONS["new_york"]
    out.extend(
        [
            ("new_york", "NY Session", ny[0], ny[1]),
            ("ny", "NY Session", ny[0], ny[1]),
        ]
    )
    asia = LIVE_FOREX_SESSIONS["asia"]
    out.append(("asian", "Asian Session", asia[0], asia[1]))
    tokyo = OTHER_SESSION_WINDOWS["tokyo"]
    out.append(("tokyo", "Tokyo Session", tokyo[0], tokyo[1]))
    kz_labels = {
        "london_kz": "London Killzone",
        "ny_kz": "NY Killzone",
        "asian_kz": "Asian Killzone",
    }
    for sid, win in ICT_KILLZONE_WINDOWS.items():
        out.append((sid, kz_labels.get(sid, sid), win[0], win[1]))
    return out


def session_active_unified(session_id: str, now_utc: datetime) -> bool:
    """
    Single boolean used to assert UI / forex_engine / fire-gate alignment in tests.
    Delegates to is_named_session_active for live + overlap ids.
    """
    return is_named_session_active(session_id, now_utc)
