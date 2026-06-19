"""Post-trade learning + per-setup outcome analytics."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import func

from app.gold_ai_trader.models import GoldAiDecision, GoldAiLesson, GoldAiOutcome

logger = logging.getLogger(__name__)

LEARNING_SYSTEM_PROMPT = """You are reviewing a gold day-trading AI's demo performance.
Produce a SHORT, ACTIONABLE lessons digest for ONE session (London or NY).

Rules:
- Cite specific setup types and win/loss counts from the data.
- Name concrete failure patterns (e.g. entered before reclaim close, chased extended moves).
- Give correctable rules, not generic platitudes.
- 4–8 sentences max."""


def compute_r_multiple(
    *,
    direction: str,
    entry: float,
    stop_loss: float,
    pnl_pct: float,
) -> Optional[float]:
    if entry <= 0 or stop_loss <= 0:
        return None
    risk = abs(entry - stop_loss) / entry * 100.0
    if risk <= 0:
        return None
    return float(pnl_pct) / risk


def get_setup_stats(db, *, days: int = 14) -> List[Dict[str, Any]]:
    """Per (setup_type, session): count, win rate, avg R, total pnl."""
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(GoldAiOutcome)
        .filter(GoldAiOutcome.closed_ts.isnot(None), GoldAiOutcome.closed_ts >= since)
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
                "setup_type": b["setup_type"],
                "session": b["session"],
                "trades": t,
                "win_rate": round(b["wins"] / t, 3) if t else 0.0,
                "wins": b["wins"],
                "losses": b["losses"],
                "breakevens": b["breakevens"],
                "avg_r_multiple": round(b["r_sum"] / b["r_count"], 2) if b["r_count"] else None,
                "total_pnl_pct": round(b["total_pnl"], 3),
            }
        )
    out.sort(key=lambda x: (-x["trades"], x["setup_type"]))
    return out


def _learning_due(db, session: str, cfg) -> bool:
    from app.strategy_models import StrategyExecution

    since = datetime.utcnow() - timedelta(days=7)
    closed_count = (
        db.query(func.count(StrategyExecution.id))
        .filter(
            StrategyExecution.notes.like("%gold_ai_trader%"),
            ~StrategyExecution.notes.like("%live_mirror%"),
            StrategyExecution.closed_at.isnot(None),
            StrategyExecution.closed_at >= since,
        )
        .scalar()
        or 0
    )
    if closed_count < cfg.learning_every_n_closes:
        return False

    now = datetime.utcnow()
    if getattr(cfg, "learning_daily_at_ny_end", True):
        if now.hour >= int(cfg.ny_end_hour):
            today = now.date()
            recent = (
                db.query(GoldAiLesson)
                .filter(
                    GoldAiLesson.session == session,
                    GoldAiLesson.ts >= datetime.combine(today, datetime.min.time()),
                )
                .first()
            )
            return recent is None
        return False

    # N-closes mode: run if no lesson in last 24h for session
    recent = (
        db.query(GoldAiLesson)
        .filter(
            GoldAiLesson.session == session,
            GoldAiLesson.ts >= now - timedelta(hours=24),
        )
        .first()
    )
    return recent is None


def _build_learning_prompt(db, session: str) -> str:
    stats = get_setup_stats(db, days=14)
    session_stats = [s for s in stats if s["session"] == session] or stats

    lines = [f"Session under review: {session.upper()}", "", "=== SETUP STATS (14d) ==="]
    for s in session_stats:
        lines.append(
            f"- {s['setup_type']} ({s['session']}): {s['trades']} trades, "
            f"win rate {s['win_rate']*100:.0f}%, avg R {s['avg_r_multiple']}, "
            f"total pnl {s['total_pnl_pct']:+.2f}%"
        )

    since = datetime.utcnow() - timedelta(days=7)
    outcomes = (
        db.query(GoldAiOutcome, GoldAiDecision)
        .join(GoldAiDecision, GoldAiDecision.id == GoldAiOutcome.decision_id)
        .filter(GoldAiOutcome.closed_ts >= since, GoldAiDecision.session == session)
        .order_by(GoldAiOutcome.closed_ts.desc())
        .limit(12)
        .all()
    )
    lines.append("")
    lines.append("=== RECENT CLOSED TRADES ===")
    for out, dec in outcomes:
        d = dec.decision or {}
        lines.append(
            f"- {out.closed_ts} setup={out.setup_type} result={out.result} "
            f"pnl={out.pnl:+.2f}% R={out.r_multiple} dir={d.get('direction')} "
            f"entry={d.get('entry')} sl={d.get('stop_loss')} "
            f"conf={dec.confidence} rationale={(d.get('rationale') or '')[:120]}"
        )

    lines.append("")
    lines.append(
        "Write the lessons digest for this session only. Be specific about "
        "which setup types worked/failed and what entry rule to tighten."
    )
    return "\n".join(lines)


async def maybe_run_learning_review(db, session: str, cfg) -> Optional[str]:
    if not _learning_due(db, session, cfg):
        return None

    prompt = _build_learning_prompt(db, session)
    try:
        import anthropic
        import os

        from app.gold_ai_trader.claude import _estimate_cost

        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
            "AI_INTEGRATIONS_ANTHROPIC_API_KEY"
        )
        if not key:
            return None
        client = anthropic.AsyncAnthropic(api_key=key)
        msg = await client.messages.create(
            model=getattr(cfg, "model", "claude-opus-4-8"),
            max_tokens=500,
            system=[
                {
                    "type": "text",
                    "text": LEARNING_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        digest = (msg.content[0].text or "").strip() if msg.content else ""
        if not digest:
            return None
        usage = msg.usage
        tin, tout, _, _, cost = _estimate_cost(usage)
        row = GoldAiLesson(
            session=session,
            digest=digest,
            tokens_in=tin,
            tokens_out=tout,
            cost_usd=cost,
        )
        db.add(row)
        db.commit()
        logger.info(
            "[gold-ai-trader] lessons updated session=%s cost=$%.4f", session, cost
        )
        return digest
    except Exception as e:
        logger.warning("[gold-ai-trader] learning review failed: %s", e)
        return None


def record_outcome_from_execution(db, decision_id: int, execution) -> bool:
    """Persist outcome from a closed execution. Returns True if newly recorded."""
    if not execution or execution.outcome == "OPEN":
        return False
    existing = (
        db.query(GoldAiOutcome)
        .filter(GoldAiOutcome.decision_id == decision_id)
        .first()
    )
    if existing:
        return False

    dec = db.query(GoldAiDecision).filter(GoldAiDecision.id == decision_id).first()
    d = (dec.decision or {}) if dec else {}
    entry = float(d.get("entry") or execution.entry_price or 0)
    sl = float(d.get("stop_loss") or execution.sl_price or 0)
    direction = (d.get("direction") or execution.direction or "long").upper()

    result = "breakeven"
    if execution.outcome == "WIN":
        result = "win"
    elif execution.outcome == "LOSS":
        result = "loss"

    pnl = float(execution.pnl_pct or 0)
    r_mult = compute_r_multiple(
        direction=direction,
        entry=entry,
        stop_loss=sl,
        pnl_pct=pnl,
    )

    row = GoldAiOutcome(
        decision_id=decision_id,
        setup_type=getattr(dec, "candidate_type", None) if dec else None,
        session=getattr(dec, "session", None) if dec else None,
        result=result,
        pnl=pnl,
        r_multiple=r_mult,
        mfe=float(execution.mfe_pips or 0),
        mae=float(execution.mae_pips or 0),
        closed_ts=execution.closed_at,
    )
    db.add(row)
    db.commit()
    return True
