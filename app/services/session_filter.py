"""UTC session windows for forex entry filtering."""
from __future__ import annotations

from datetime import datetime
from typing import Dict, Tuple

SESSIONS: Dict[str, Tuple[int, int, int, int]] = {
    "sydney": (21, 0, 6, 0),
    "tokyo": (0, 0, 9, 0),
    # Asia live-forex confirm window — fixed UTC (01:00–04:00 = 02:00–05:00 UK BST).
    "asia": (1, 0, 4, 0),
    "asian": (1, 0, 4, 0),
    "asia_kz": (1, 0, 4, 0),
    "london": (7, 0, 16, 0),
    "newyork": (12, 0, 21, 0),
    "london_kz": (7, 0, 10, 0),
    "ny_kz": (12, 0, 15, 0),
}


def _in_window(now: datetime, start_h: int, start_m: int, end_h: int, end_m: int) -> bool:
    t = now.hour * 60 + now.minute
    start = start_h * 60 + start_m
    end = end_h * 60 + end_m
    if start > end:
        return t >= start or t < end
    return start <= t < end


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
        win = SESSIONS.get(sid.replace("new_york", "newyork").replace("-", "_"))
        if not win:
            continue
        if _in_window(now_utc, win[0], win[1], win[2], win[3]):
            return True, ""
    return False, "session_filter"
