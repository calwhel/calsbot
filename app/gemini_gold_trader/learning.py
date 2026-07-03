"""Per-setup outcome analytics for Gemini Gold calibration."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldOutcome


def get_setup_stats(db, *, days: int = 14) -> List[Dict[str, Any]]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(GeminiGoldOutcome)
        .filter(GeminiGoldOutcome.closed_ts.isnot(None), GeminiGoldOutcome.closed_ts >= since)
        .all()
    )
    buckets: Dict[tuple, Dict[str, Any]] = {}
    for o in rows:
        key = (o.setup_type or "unknown", o.session or "unknown")
        b = buckets.setdefault(
            key,
            {
                "setup_type": key[0],
                "session": key[1],
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "breakevens": 0,
                "total_pnl": 0.0,
                "r_sum": 0.0,
                "r_count": 0,
            },
        )
        b["trades"] += 1
        if o.result == "win":
            b["wins"] += 1
        elif o.result == "loss":
            b["losses"] += 1
        else:
            b["breakevens"] += 1
        b["total_pnl"] += float(o.pnl or 0)
        if o.r_multiple is not None:
            b["r_sum"] += float(o.r_multiple)
            b["r_count"] += 1

    out = []
    for b in buckets.values():
        t = b["trades"]
        out.append(
            {
                **b,
                "win_rate": round(100.0 * b["wins"] / t, 1) if t else 0.0,
                "avg_r": round(b["r_sum"] / b["r_count"], 2) if b["r_count"] else None,
            }
        )
    out.sort(key=lambda x: (-x["trades"], x["setup_type"]))
    return out


def call_stats_today(db) -> List[Dict[str, Any]]:
    today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = (
        db.query(GeminiGoldDecision)
        .filter(GeminiGoldDecision.ts >= today)
        .all()
    )
    buckets: Dict[str, Dict[str, int]] = {}
    for d in rows:
        setup = "unknown"
        if isinstance(d.decision, dict):
            setup = str(d.decision.get("setup_type") or "unknown")
        b = buckets.setdefault(setup, {"setup_type": setup, "calls": 0, "takes": 0, "executed": 0})
        b["calls"] += 1
        if (d.action or "").upper() == "TAKE":
            b["takes"] += 1
        if d.executed:
            b["executed"] += 1
    return sorted(buckets.values(), key=lambda x: -x["calls"])
