"""Telegram alerts for Gold AI Trader (demo-labelled, gold module only)."""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import date, datetime
from typing import Any, Dict, Optional

from app.gold_ai_trader.config import SYMBOL, GoldAiRuntimeConfig

logger = logging.getLogger(__name__)

_PREFIX = "[DEMO] Gold AI Trader"
_last_daily_summary_date: Optional[date] = None


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def telegram_notifications_enabled() -> bool:
    return _env_bool("GOLD_AI_TRADER_TELEGRAM", True)


def daily_summary_enabled() -> bool:
    return _env_bool("GOLD_AI_TRADER_TELEGRAM_DAILY", True)


def _html_escape(text: str) -> str:
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _fmt_price(v) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


def resolve_notify_chat_id(db, user_id: Optional[int]) -> Optional[str]:
    """Trader user's Telegram chat, else owner fallback."""
    if user_id:
        from app.models import User

        u = db.query(User).filter(User.id == user_id).first()
        if u and getattr(u, "telegram_id", None):
            return str(u.telegram_id).strip()
    try:
        from app.services.telegram_dm import owner_chat_id

        return owner_chat_id()
    except Exception:
        return os.getenv("OWNER_TELEGRAM_ID")


async def _send(text: str, *, msg_type: str = "gold_ai_trader", exec_id: int = 0) -> bool:
    if not telegram_notifications_enabled():
        return False
    try:
        from app.services.telegram_dm import send_dm, bot_tokens_for_asset

        db = __import__("app.database", fromlist=["SessionLocal"]).SessionLocal()
        try:
            from app.gold_ai_trader.config import env_defaults
            from app.gold_ai_trader.guardrails import merge_config
            from app.gold_ai_trader.schema import seed_config_if_missing

            row = seed_config_if_missing(db)
            cfg = merge_config(row, env_defaults())
            chat_id = resolve_notify_chat_id(db, cfg.demo_user_id)
        finally:
            db.close()
        if not chat_id:
            logger.debug("[gold-ai-trader] telegram skipped: no chat id")
            return False
        return await send_dm(
            chat_id,
            text,
            parse_mode="HTML",
            tokens=bot_tokens_for_asset("forex"),
            msg_type=msg_type,
            symbol=SYMBOL,
            exec_id=exec_id,
        )
    except Exception as e:
        logger.warning("[gold-ai-trader] telegram send failed: %s", e)
        return False


def format_take_message(
    *,
    candidate_type: str,
    session: str,
    decision: Dict[str, Any],
    confidence: int,
    executed: bool,
    execution_id: Optional[int] = None,
    block_reason: Optional[str] = None,
) -> str:
    direction = (decision.get("direction") or "—").upper()
    rationale = _html_escape(decision.get("rationale") or "—")
    setup = _html_escape(candidate_type or "unknown")
    sess = _html_escape(session or "—")
    lines = [
        f"<b>{_PREFIX} — TAKE</b>",
        f"Setup: <b>{setup}</b> · {sess}",
        f"Direction: <b>{direction}</b>",
        (
            f"Entry: {_fmt_price(decision.get('entry'))} | "
            f"SL: {_fmt_price(decision.get('stop_loss'))} | "
            f"TP: {_fmt_price(decision.get('take_profit'))}"
        ),
        f"Confidence: <b>{confidence}%</b>",
        f"Rationale: {rationale}",
    ]
    if executed and execution_id:
        lines.append(f"Status: ✅ Demo order placed (exec #{execution_id})")
    elif block_reason:
        lines.append(f"Status: ⚠️ Not executed — {_html_escape(block_reason)}")
    else:
        lines.append("Status: ⚠️ Not executed")
    return "\n".join(lines)


def format_close_message(
    *,
    candidate_type: str,
    session: str,
    direction: str,
    outcome: str,
    pnl_pct: float,
    pnl_usd: Optional[float],
    decision_id: int,
    execution_id: int,
) -> str:
    result = outcome.upper() if outcome else "CLOSED"
    pnl_line = f"P&amp;L: {pnl_pct:+.2f}%"
    if pnl_usd is not None:
        pnl_line += f" (${pnl_usd:+.2f})"
    return "\n".join(
        [
            f"<b>{_PREFIX} — CLOSED {result}</b>",
            f"{SYMBOL} {direction.upper()} · {_html_escape(candidate_type or 'setup')}",
            f"Session: {_html_escape(session or '—')} · Decision #{decision_id}",
            pnl_line,
            f"Exec #{execution_id}",
        ]
    )


def format_daily_summary(
    *,
    calls: int,
    max_calls: int,
    trades: int,
    max_trades: int,
    cost_usd: float,
    demo_pnl_usd: float,
    open_positions: int,
) -> str:
    return "\n".join(
        [
            f"<b>{_PREFIX} — Daily summary</b>",
            f"Claude calls: {calls}/{max_calls}",
            f"Trades: {trades}/{max_trades}",
            f"API cost: ${cost_usd:.2f}",
            f"Demo P&amp;L today: ${demo_pnl_usd:+.2f}",
            f"Open demo positions: {open_positions}",
        ]
    )


async def notify_take_decision(
    *,
    candidate_type: str,
    session: str,
    decision: Dict[str, Any],
    confidence: int,
    executed: bool,
    execution_id: Optional[int] = None,
    block_reason: Optional[str] = None,
) -> bool:
    text = format_take_message(
        candidate_type=candidate_type,
        session=session,
        decision=decision,
        confidence=confidence,
        executed=executed,
        execution_id=execution_id,
        block_reason=block_reason,
    )
    return await _send(text, msg_type="gold_ai_take", exec_id=int(execution_id or 0))


async def notify_trade_close(
    *,
    candidate_type: str,
    session: str,
    direction: str,
    outcome: str,
    pnl_pct: float,
    pnl_usd: Optional[float],
    decision_id: int,
    execution_id: int,
) -> bool:
    text = format_close_message(
        candidate_type=candidate_type,
        session=session,
        direction=direction,
        outcome=outcome,
        pnl_pct=pnl_pct,
        pnl_usd=pnl_usd,
        decision_id=decision_id,
        execution_id=execution_id,
    )
    return await _send(text, msg_type="gold_ai_close", exec_id=execution_id)


def _decision_id_from_notes(notes: str) -> Optional[int]:
    if not notes:
        return None
    m = re.search(r"decision_id=(\d+)", notes)
    return int(m.group(1)) if m else None


async def sync_closed_trade_notifications(db, cfg: GoldAiRuntimeConfig) -> int:
    """Record missing outcomes and notify on newly closed demo trades."""
    from app.strategy_models import StrategyExecution
    from app.gold_ai_trader.models import GoldAiDecision, GoldAiOutcome
    from app.gold_ai_trader.learning import record_outcome_from_execution

    if not telegram_notifications_enabled():
        return 0

    since = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    closed = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.notes.like("%gold_ai_trader%"),
            ~StrategyExecution.notes.like("%live_mirror%"),
            StrategyExecution.outcome != "OPEN",
            StrategyExecution.closed_at.isnot(None),
            StrategyExecution.closed_at >= since - __import__("datetime").timedelta(days=3),
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(30)
        .all()
    )
    sent = 0
    for ex in closed:
        did = _decision_id_from_notes(ex.notes or "")
        if not did:
            continue
        was_new = record_outcome_from_execution(db, did, ex)
        if not was_new:
            continue
        dec = db.query(GoldAiDecision).filter(GoldAiDecision.id == did).first()
        ok = await notify_trade_close(
            candidate_type=getattr(dec, "candidate_type", None) or "?",
            session=getattr(dec, "session", None) or "—",
            direction=ex.direction or "?",
            outcome=ex.outcome or "?",
            pnl_pct=float(ex.pnl_pct or 0),
            pnl_usd=float(ex.pnl_usd) if ex.pnl_usd is not None else None,
            decision_id=did,
            execution_id=ex.id,
        )
        if ok:
            sent += 1
    return sent


async def maybe_send_daily_summary(db, cfg: GoldAiRuntimeConfig) -> bool:
    global _last_daily_summary_date
    if not daily_summary_enabled() or not telegram_notifications_enabled():
        return False
    today = datetime.utcnow().date()
    if _last_daily_summary_date == today:
        return False
    # Send once per UTC day after NY session window (16:00 UTC)
    if datetime.utcnow().hour < int(cfg.ny_end_hour):
        return False
    from app.gold_ai_trader.guardrails import (
        calls_today,
        trades_today,
        cost_today_usd,
        demo_pnl_today_usd,
        open_position_count,
    )

    text = format_daily_summary(
        calls=calls_today(db),
        max_calls=cfg.max_calls_day,
        trades=trades_today(db),
        max_trades=cfg.max_trades_day,
        cost_usd=cost_today_usd(db),
        demo_pnl_usd=demo_pnl_today_usd(db, cfg.demo_user_id or 0),
        open_positions=open_position_count(db, cfg.demo_user_id or 0),
    )
    ok = await _send(text, msg_type="gold_ai_daily")
    if ok:
        _last_daily_summary_date = today
    return ok
