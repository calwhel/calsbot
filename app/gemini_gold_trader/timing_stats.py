"""UTC hour / session performance stats for Gemini Gold."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List

from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldOutcome


def _win_rate(wins: int, trades: int) -> float:
    return round(100.0 * wins / trades, 1) if trades else 0.0


def hour_performance_stats(db, *, days: int = 14, min_trades: int = 1) -> Dict[str, Any]:
    """Win rate by UTC hour from executed closed trades + take volume from all decisions."""
    since = datetime.utcnow() - timedelta(days=days)

    closed_rows = (
        db.query(GeminiGoldOutcome, GeminiGoldDecision)
        .join(GeminiGoldDecision, GeminiGoldDecision.id == GeminiGoldOutcome.decision_id)
        .filter(
            GeminiGoldOutcome.closed_ts.isnot(None),
            GeminiGoldOutcome.closed_ts >= since,
            GeminiGoldDecision.executed.is_(True),
        )
        .all()
    )

    hour_buckets: Dict[int, Dict[str, Any]] = {}
    session_buckets: Dict[str, Dict[str, Any]] = {}

    for out, dec in closed_rows:
        ts = dec.ts or out.closed_ts
        if not ts:
            continue
        h = int(ts.hour)
        b = hour_buckets.setdefault(
            h,
            {"hour_utc": h, "trades": 0, "wins": 0, "losses": 0, "breakevens": 0},
        )
        b["trades"] += 1
        if out.result == "win":
            b["wins"] += 1
        elif out.result == "loss":
            b["losses"] += 1
        else:
            b["breakevens"] += 1

        sess = (out.session or dec.session or "unknown").lower()
        sb = session_buckets.setdefault(
            sess,
            {"session": sess, "trades": 0, "wins": 0, "losses": 0},
        )
        sb["trades"] += 1
        if out.result == "win":
            sb["wins"] += 1
        elif out.result == "loss":
            sb["losses"] += 1

    take_rows = (
        db.query(GeminiGoldDecision)
        .filter(
            GeminiGoldDecision.ts >= since,
            GeminiGoldDecision.action == "TAKE",
        )
        .all()
    )
    take_by_hour: Dict[int, int] = {}
    for dec in take_rows:
        if not dec.ts:
            continue
        h = int(dec.ts.hour)
        take_by_hour[h] = take_by_hour.get(h, 0) + 1

    by_hour: List[Dict[str, Any]] = []
    for h in sorted(hour_buckets):
        b = hour_buckets[h]
        trades = int(b["trades"])
        b["win_rate_pct"] = _win_rate(int(b["wins"]), trades)
        b["take_count"] = int(take_by_hour.get(h, 0))
        by_hour.append(b)

    for h, count in take_by_hour.items():
        if h not in hour_buckets:
            by_hour.append(
                {
                    "hour_utc": h,
                    "trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "breakevens": 0,
                    "win_rate_pct": 0.0,
                    "take_count": count,
                }
            )
    by_hour.sort(key=lambda x: int(x["hour_utc"]))

    qualified = [b for b in by_hour if int(b["trades"]) >= min_trades]
    best = sorted(
        qualified,
        key=lambda x: (-float(x["win_rate_pct"]), -int(x["trades"]), int(x["hour_utc"])),
    )[:5]
    worst = sorted(
        qualified,
        key=lambda x: (float(x["win_rate_pct"]), -int(x["trades"]), int(x["hour_utc"])),
    )[:5]

    by_session: List[Dict[str, Any]] = []
    for sess in sorted(session_buckets):
        b = session_buckets[sess]
        trades = int(b["trades"])
        by_session.append(
            {
                **b,
                "win_rate_pct": _win_rate(int(b["wins"]), trades),
            }
        )
    by_session.sort(key=lambda x: (-float(x.get("win_rate_pct") or 0), -int(x["trades"])))

    return {
        "days": days,
        "by_hour": by_hour,
        "by_session": by_session,
        "best_hours": best,
        "worst_hours": worst,
        "total_closed_trades": sum(int(b["trades"]) for b in by_hour),
    }
