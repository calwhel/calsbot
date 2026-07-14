"""Configurable trading windows for Gemini Gold (sessions + optional UTC hours)."""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig
from app.services.forex_sessions import (
    LIVE_FOREX_SESSIONS,
    format_window_clock,
    in_window,
    is_named_session_active,
)

DEFAULT_TRADE_SESSIONS: Tuple[str, ...] = ("asia", "london", "new_york")
_SESSION_ORDER: Tuple[str, ...] = ("asia", "new_york", "london")
_HHMM_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def normalize_trade_sessions(raw: Any) -> Tuple[str, ...]:
    """Return enabled session ids (asia, london, new_york)."""
    if raw is None:
        return DEFAULT_TRADE_SESSIONS
    if isinstance(raw, str):
        parts = [p.strip().lower() for p in raw.replace("|", ",").split(",") if p.strip()]
    elif isinstance(raw, (list, tuple, set)):
        parts = [str(p).strip().lower() for p in raw if str(p).strip()]
    else:
        return DEFAULT_TRADE_SESSIONS
    out: List[str] = []
    for p in parts:
        if p in ("asian", "asia"):
            key = "asia"
        elif p in ("ny", "newyork", "new_york", "us"):
            key = "new_york"
        elif p in ("london", "europe"):
            key = "london"
        else:
            key = p
        if key in DEFAULT_TRADE_SESSIONS and key not in out:
            out.append(key)
    return tuple(out) if out else DEFAULT_TRADE_SESSIONS


def parse_hhmm(value: Any, *, default: str = "00:00") -> Tuple[int, int]:
    raw = str(value or default).strip()
    m = _HHMM_RE.match(raw)
    if not m:
        return 0, 0
    h = max(0, min(23, int(m.group(1))))
    minute = max(0, min(59, int(m.group(2))))
    return h, minute


def format_hhmm(h: int, m: int) -> str:
    return f"{int(h):02d}:{int(m):02d}"


def in_custom_trade_window(now: datetime, start: str, end: str) -> bool:
    sh, sm = parse_hhmm(start, default="00:00")
    eh, em = parse_hhmm(end, default="23:59")
    return in_window(now, sh, sm, eh, em)


def trade_session_catalog() -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    labels = {
        "asia": "Asia",
        "london": "London",
        "new_york": "US / New York",
    }
    for sid in DEFAULT_TRADE_SESSIONS:
        win = LIVE_FOREX_SESSIONS[sid]
        out.append(
            {
                "id": sid,
                "label": labels.get(sid, sid),
                "clock": f"{format_window_clock(win)} UTC",
            }
        )
    return out


def trade_schedule_from_cfg(cfg: GeminiGoldRuntimeConfig) -> Dict[str, Any]:
    sessions = list(normalize_trade_sessions(getattr(cfg, "trade_sessions", None)))
    custom = bool(getattr(cfg, "custom_trade_hours_enabled", False))
    start = format_hhmm(*parse_hhmm(getattr(cfg, "trade_hours_start_utc", None), default="12:00"))
    end = format_hhmm(*parse_hhmm(getattr(cfg, "trade_hours_end_utc", None), default="21:00"))
    return {
        "trade_sessions": sessions,
        "custom_trade_hours_enabled": custom,
        "trade_hours_start_utc": start,
        "trade_hours_end_utc": end,
    }


def trade_schedule_summary(cfg: GeminiGoldRuntimeConfig) -> str:
    sched = trade_schedule_from_cfg(cfg)
    sess = ", ".join(sched["trade_sessions"]) or "none"
    if sched["custom_trade_hours_enabled"]:
        return (
            f"sessions=[{sess}] AND custom UTC {sched['trade_hours_start_utc']}"
            f"–{sched['trade_hours_end_utc']}"
        )
    return f"sessions=[{sess}]"


def resolve_trading_session(
    now: datetime,
    cfg: GeminiGoldRuntimeConfig,
) -> Tuple[Optional[str], str]:
    """
    Return (active_session, dormant_reason).
    dormant_reason is ok when session is set.
    """
    enabled = set(normalize_trade_sessions(getattr(cfg, "trade_sessions", None)))
    active: Optional[str] = None
    for sid in _SESSION_ORDER:
        if sid not in enabled:
            continue
        if is_named_session_active(sid, now):
            active = sid
            break

    if not active:
        return None, "outside_session"

    if bool(getattr(cfg, "custom_trade_hours_enabled", False)):
        start = getattr(cfg, "trade_hours_start_utc", None) or "12:00"
        end = getattr(cfg, "trade_hours_end_utc", None) or "21:00"
        if not in_custom_trade_window(now, str(start), str(end)):
            return None, "outside_trade_hours"

    return active, "ok"


def trading_allowed_now(now: datetime, cfg: GeminiGoldRuntimeConfig) -> bool:
    session, _ = resolve_trading_session(now, cfg)
    return session is not None
