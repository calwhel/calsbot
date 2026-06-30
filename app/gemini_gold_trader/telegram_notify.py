"""Telegram alerts for Gemini Gold Trader."""
from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any, Dict, Optional

from app.gemini_gold_trader.config import SYMBOL, GeminiGoldRuntimeConfig

logger = logging.getLogger(__name__)

_PREFIX = "[Gemini Gold]"
_last_call_cap_notify_date: Optional[date] = None


def clear_call_cap_notify_state() -> None:
    global _last_call_cap_notify_date
    _last_call_cap_notify_date = None


def _env_bool(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def telegram_notifications_enabled() -> bool:
    return _env_bool("GEMINI_GOLD_TELEGRAM", True)


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


async def _send(text: str, *, msg_type: str = "gemini_gold_trader", exec_id: int = 0) -> bool:
    if not telegram_notifications_enabled():
        return False
    try:
        from app.services.telegram_dm import bot_tokens_for_asset, send_dm
        from app.gemini_gold_trader.config import env_defaults
        from app.gemini_gold_trader.guardrails import merge_config
        from app.gemini_gold_trader.schema import seed_config_if_missing
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            row = seed_config_if_missing(db)
            cfg = merge_config(row, env_defaults())
            chat_id = resolve_notify_chat_id(db, cfg.demo_user_id)
        finally:
            db.close()
        if not chat_id:
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
    except Exception as exc:
        logger.warning("[gemini-gold] telegram send failed: %s", exc)
        return False


def format_decision_message(
    *,
    session: str,
    decision: Dict[str, Any],
    action: str,
    confidence: int,
    executed: bool = False,
    execution_id: Optional[int] = None,
    block_reason: Optional[str] = None,
    dry_run: bool = False,
) -> str:
    direction = (decision.get("direction") or "—").upper()
    rationale = _html_escape(decision.get("rationale") or "—")
    sess = _html_escape(session or "—")
    action_up = (action or "SKIP").upper()
    title = f"{_PREFIX} — {action_up}"
    if dry_run and action_up == "TAKE":
        title = f"{_PREFIX} — [DRY-RUN] would TAKE"
    lines = [
        f"<b>{title}</b>",
        f"Session: {sess}",
    ]
    if action_up == "TAKE":
        lines.extend(
            [
                f"Direction: <b>{direction}</b>",
                (
                    f"Entry: {_fmt_price(decision.get('entry'))} | "
                    f"SL: {_fmt_price(decision.get('stop_loss'))} | "
                    f"TP: {_fmt_price(decision.get('take_profit'))}"
                ),
                f"Confidence: <b>{confidence}%</b>",
            ]
        )
    else:
        lines.append(f"Confidence: <b>{confidence}%</b>")
    lines.append(f"Rationale: {rationale}")
    if executed and execution_id:
        lines.append(f"Status: ✅ Demo order placed (exec #{execution_id})")
    elif block_reason:
        lines.append(f"Status: ⚠️ Not executed — {_html_escape(block_reason)}")
    elif dry_run and action_up == "TAKE":
        lines.append("Status: 🔍 Dry-run — no broker order")
    return "\n".join(lines)


async def notify_decision(
    *,
    session: str,
    decision: Dict[str, Any],
    action: str,
    confidence: int,
    executed: bool = False,
    execution_id: Optional[int] = None,
    block_reason: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    text = format_decision_message(
        session=session,
        decision=decision,
        action=action,
        confidence=confidence,
        executed=executed,
        execution_id=execution_id,
        block_reason=block_reason,
        dry_run=dry_run,
    )
    await _send(text, msg_type="gemini_gold_decision", exec_id=execution_id or 0)


async def maybe_notify_call_cap_reached() -> None:
    global _last_call_cap_notify_date
    today = date.today()
    if _last_call_cap_notify_date == today:
        return
    _last_call_cap_notify_date = today
    await _send(
        f"<b>{_PREFIX} — daily Gemini cap reached</b>\n"
        "Scan loop dormant until UTC midnight or manual reset.",
        msg_type="gemini_gold_call_cap",
    )


async def notify_trade_closed(
    *,
    session: str,
    direction: str,
    outcome: str,
    pnl_pct: float,
    decision_id: int,
    execution_id: int,
) -> None:
    result = outcome.upper()
    text = (
        f"<b>{_PREFIX} — CLOSED {result}</b>\n"
        f"Session: {_html_escape(session)} · {direction.upper()}\n"
        f"PnL: {pnl_pct:+.2f}% · decision #{decision_id} · exec #{execution_id}"
    )
    await _send(text, msg_type="gemini_gold_close", exec_id=execution_id)
