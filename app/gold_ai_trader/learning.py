"""Post-trade learning digest (periodic Claude call)."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from app.gold_ai_trader.models import GoldAiDecision, GoldAiLesson, GoldAiOutcome

logger = logging.getLogger(__name__)


async def maybe_run_learning_review(db, session: str, cfg) -> Optional[str]:
    """After N closed trades, compile lessons via Claude (Haiku/Opus — use configured model)."""
    from app.strategy_models import StrategyExecution

    since = datetime.utcnow() - timedelta(days=7)
    closed = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.notes.like("%gold_ai_trader%"),
            ~StrategyExecution.notes.like("%live_mirror%"),
            StrategyExecution.closed_at.isnot(None),
            StrategyExecution.closed_at >= since,
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(20)
        .all()
    )
    if len(closed) < cfg.learning_every_n_closes:
        return None

    recent_decisions = (
        db.query(GoldAiDecision)
        .filter(GoldAiDecision.ts >= since)
        .order_by(GoldAiDecision.ts.desc())
        .limit(30)
        .all()
    )
    lines = ["Recent Gold AI Trader activity:"]
    for d in recent_decisions[:15]:
        lines.append(
            f"- {d.ts} session={d.session} action={d.action} conf={d.confidence} "
            f"executed={d.executed} type={d.candidate_type}"
        )
    for ex in closed[:10]:
        lines.append(
            f"- TRADE {ex.direction} outcome={ex.outcome} pnl={ex.pnl_pct}% "
            f"mfe={ex.mfe_pips} mae={ex.mae_pips}"
        )

    prompt = (
        "You are reviewing a gold day-trading AI's recent demo performance. "
        "Produce a 4-6 sentence digest of what is working vs failing, "
        "split by London vs NY if possible. Be specific and actionable.\n\n"
        + "\n".join(lines)
    )

    try:
        import anthropic
        import os

        key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
            "AI_INTEGRATIONS_ANTHROPIC_API_KEY"
        )
        if not key:
            return None
        client = anthropic.AsyncAnthropic(api_key=key)
        msg = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        digest = (msg.content[0].text or "").strip() if msg.content else ""
        if not digest:
            return None
        row = GoldAiLesson(session=session, digest=digest)
        db.add(row)
        db.commit()
        logger.info("[gold-ai-trader] lessons updated for %s", session)
        return digest
    except Exception as e:
        logger.warning("[gold-ai-trader] learning review failed: %s", e)
        return None


def record_outcome_from_execution(db, decision_id: int, execution) -> None:
    if not execution or execution.outcome == "OPEN":
        return
    existing = (
        db.query(GoldAiOutcome)
        .filter(GoldAiOutcome.decision_id == decision_id)
        .first()
    )
    if existing:
        return
    result = "breakeven"
    if execution.outcome == "WIN":
        result = "win"
    elif execution.outcome == "LOSS":
        result = "loss"
    row = GoldAiOutcome(
        decision_id=decision_id,
        result=result,
        pnl=float(execution.pnl_pct or 0),
        mfe=float(execution.mfe_pips or 0),
        mae=float(execution.mae_pips or 0),
        closed_ts=execution.closed_at,
    )
    db.add(row)
    db.commit()
