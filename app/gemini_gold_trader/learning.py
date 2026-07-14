"""Per-setup outcome analytics for Gemini Gold calibration."""
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


def get_setup_stats(
    db,
    *,
    days: int = 14,
    user_id: Optional[int] = None,
    ctrader_account_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Closed-trade stats from cTrader StrategyExecution rows (broker ground truth)."""
    since = datetime.utcnow() - timedelta(days=days)
    if not user_id:
        return []

    rows = gemini_broker_executions_query(
        db,
        user_id=int(user_id),
        since=since,
        ctrader_account_id=ctrader_account_id,
        closed_only=True,
    ).all()

    decision_ids = [did for ex in rows if (did := decision_id_from_execution(ex))]
    decision_meta: Dict[int, GeminiGoldDecision] = {}
    if decision_ids:
        for dec in (
            db.query(GeminiGoldDecision)
            .filter(GeminiGoldDecision.id.in_(decision_ids))
            .all()
        ):
            decision_meta[int(dec.id)] = dec

    buckets: Dict[tuple, Dict[str, Any]] = {}
    for ex in rows:
        dec = decision_meta.get(int(decision_id_from_execution(ex) or 0))
        setup_type = None
        if dec:
            setup_type = dec.setup_type
            if not setup_type and isinstance(dec.decision, dict):
                setup_type = dec.decision.get("setup_type")
        setup_type = str(setup_type or "unknown")
        ts = ex.fired_at or ex.closed_at
        sess = (getattr(dec, "session", None) if dec else None) or (
            active_live_forex_session(ts) if ts else None
        ) or "unknown"
        key = (setup_type, str(sess).lower())
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
        result = broker_outcome_label(ex.outcome)
        if result == "win":
            b["wins"] += 1
        elif result == "loss":
            b["losses"] += 1
        else:
            b["breakevens"] += 1
        b["total_pnl"] += float(ex.pnl_pct or 0)

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
        setup = "skip_unlabeled"
        if d.setup_type:
            setup = str(d.setup_type)
        elif isinstance(d.decision, dict):
            raw = d.decision.get("setup_type")
            if raw:
                setup = str(raw)
            elif (d.action or "").upper() != "SKIP":
                setup = "unknown"
        b = buckets.setdefault(setup, {"setup_type": setup, "calls": 0, "takes": 0, "executed": 0})
        b["calls"] += 1
        if (d.action or "").upper() == "TAKE":
            b["takes"] += 1
        if d.executed:
            b["executed"] += 1
    return sorted(buckets.values(), key=lambda x: -x["calls"])
