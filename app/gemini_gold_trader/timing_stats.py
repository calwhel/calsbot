"""UTC hour / session performance stats for Gemini Gold (cTrader broker ground truth)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from app.gemini_gold_trader.models import GeminiGoldDecision
from app.gemini_gold_trader.outcomes import (
    broker_outcome_label,
    decision_id_from_execution,
    gemini_broker_executions_query,
)
from app.services.forex_sessions import active_live_forex_session


def _win_rate(wins: int, trades: int) -> float:
    return round(100.0 * wins / trades, 1) if trades else 0.0


def _load_decision_meta(db, decision_ids: List[int]) -> Dict[int, GeminiGoldDecision]:
    if not decision_ids:
        return {}
    rows = (
        db.query(GeminiGoldDecision)
        .filter(GeminiGoldDecision.id.in_(decision_ids))
        .all()
    )
    return {int(r.id): r for r in rows}


def hour_performance_stats(
    db,
    *,
    user_id: Optional[int] = None,
    ctrader_account_id: Optional[str] = None,
    days: int = 14,
    min_trades: int = 1,
) -> Dict[str, Any]:
    """
    Win rate by UTC hour from closed cTrader demo/live executions.
    Uses entry time (fired_at) for hour buckets — same source as Closed trades panel.
    """
    since = datetime.utcnow() - timedelta(days=days)
    empty = {
        "days": days,
        "source": "ctrader_executions",
        "ctrader_account_id": ctrader_account_id,
        "by_hour": [],
        "by_session": [],
        "best_hours": [],
        "worst_hours": [],
        "total_closed_trades": 0,
    }
    if not user_id:
        return empty

    closed_rows = gemini_broker_executions_query(
        db,
        user_id=int(user_id),
        since=since,
        ctrader_account_id=ctrader_account_id,
        closed_only=True,
    ).all()

    decision_ids = [did for ex in closed_rows if (did := decision_id_from_execution(ex))]
    decision_meta = _load_decision_meta(db, decision_ids)

    hour_buckets: Dict[int, Dict[str, Any]] = {}
    session_buckets: Dict[str, Dict[str, Any]] = {}

    for ex in closed_rows:
        ts = ex.fired_at or ex.closed_at
        if not ts:
            continue
        h = int(ts.hour)
        result = broker_outcome_label(ex.outcome)
        b = hour_buckets.setdefault(
            h,
            {"hour_utc": h, "trades": 0, "wins": 0, "losses": 0, "breakevens": 0},
        )
        b["trades"] += 1
        if result == "win":
            b["wins"] += 1
        elif result == "loss":
            b["losses"] += 1
        else:
            b["breakevens"] += 1

        dec = decision_meta.get(int(decision_id_from_execution(ex) or 0))
        sess = (getattr(dec, "session", None) if dec else None) or active_live_forex_session(ts) or "unknown"
        sess = str(sess).lower()
        sb = session_buckets.setdefault(
            sess,
            {"session": sess, "trades": 0, "wins": 0, "losses": 0, "breakevens": 0},
        )
        sb["trades"] += 1
        if result == "win":
            sb["wins"] += 1
        elif result == "loss":
            sb["losses"] += 1
        else:
            sb["breakevens"] += 1

    fired_rows = gemini_broker_executions_query(
        db,
        user_id=int(user_id),
        since=since,
        ctrader_account_id=ctrader_account_id,
        closed_only=False,
    ).all()
    take_by_hour: Dict[int, int] = {}
    for ex in fired_rows:
        ts = ex.fired_at
        if not ts:
            continue
        h = int(ts.hour)
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

    total_wins = sum(int(b["wins"]) for b in by_hour)
    total_trades = sum(int(b["trades"]) for b in by_hour)

    return {
        "days": days,
        "source": "ctrader_executions",
        "ctrader_account_id": ctrader_account_id,
        "by_hour": by_hour,
        "by_session": by_session,
        "best_hours": best,
        "worst_hours": worst,
        "total_closed_trades": total_trades,
        "overall_win_rate_pct": _win_rate(total_wins, total_trades),
    }
