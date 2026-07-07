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
        from app.services.telegram_dm import bot_tokens_for_asset, deliver_trade_telegram
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
            logger.warning("[gemini-gold] telegram skipped: no chat_id for user")
            return False
        ok = await deliver_trade_telegram(
            chat_id,
            text,
            parse_mode="HTML",
            tokens=bot_tokens_for_asset("forex"),
            msg_type=msg_type,
            symbol=SYMBOL,
            exec_id=exec_id,
            asset_class="forex",
        )
        if not ok:
            logger.warning(
                "[gemini-gold] telegram delivery failed msg_type=%s exec_id=%s",
                msg_type,
                exec_id,
            )
        return ok
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
    execution_mode: str = "demo",
    fill_kind: Optional[str] = None,
) -> str:
    direction = (decision.get("direction") or "—").upper()
    rationale = _html_escape(decision.get("rationale") or "—")
    sess = _html_escape(session or "—")
    setup = str(decision.get("setup_type") or "").strip()
    action_up = (action or "SKIP").upper()
    title = f"{_PREFIX} — {action_up}"
    if dry_run and action_up == "TAKE":
        title = f"{_PREFIX} — [DRY-RUN] would TAKE"
    lines = [
        f"<b>{title}</b>",
        f"Session: {sess}",
    ]
    if setup:
        lines.append(f"Setup: <b>{_html_escape(setup)}</b>")
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
        mode = (execution_mode or "demo").lower()
        if fill_kind == "entry_watch":
            status = f"✅ Entry-watch filled ({mode} exec #{execution_id})"
        elif mode == "live":
            status = f"✅ Live order placed (exec #{execution_id})"
        else:
            status = f"✅ Demo order placed (exec #{execution_id})"
        lines.append(f"Status: {status}")
    elif block_reason:
        lines.append(f"Status: ⚠️ Not executed — {_html_escape(block_reason)}")
    elif dry_run and action_up == "TAKE":
        lines.append("Status: 🔍 Dry-run — no broker order")
    return "\n".join(lines)


async def notify_live_mirror_filled(
    *,
    session: str,
    decision: Dict[str, Any],
    confidence: int,
    decision_id: int,
    live_execution_id: int,
    demo_execution_id: int,
) -> bool:
    """Alert when live mirror copies a demo fill to the live account."""
    direction = (decision.get("direction") or "—").upper()
    setup = str(decision.get("setup_type") or "").strip()
    lines = [
        f"<b>{_PREFIX} — LIVE MIRROR</b>",
        f"Session: {_html_escape(session or '—')}",
    ]
    if setup:
        lines.append(f"Setup: <b>{_html_escape(setup)}</b>")
    lines.extend(
        [
            f"Direction: <b>{direction}</b>",
            (
                f"Entry: {_fmt_price(decision.get('entry'))} | "
                f"SL: {_fmt_price(decision.get('stop_loss'))} | "
                f"TP: {_fmt_price(decision.get('take_profit'))}"
            ),
            f"Confidence: <b>{confidence}%</b>",
            (
                f"Status: ✅ Live mirror order placed "
                f"(live exec #{live_execution_id}, demo exec #{demo_execution_id}, "
                f"decision #{decision_id})"
            ),
        ]
    )
    return await _send(
        "\n".join(lines),
        msg_type="gemini_gold_live_mirror",
        exec_id=live_execution_id,
    )


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
    execution_mode: str = "demo",
    fill_kind: Optional[str] = None,
) -> bool:
    text = format_decision_message(
        session=session,
        decision=decision,
        action=action,
        confidence=confidence,
        executed=executed,
        execution_id=execution_id,
        block_reason=block_reason,
        dry_run=dry_run,
        execution_mode=execution_mode,
        fill_kind=fill_kind,
    )
    return await _send(text, msg_type="gemini_gold_decision", exec_id=execution_id or 0)


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
) -> bool:
    result = outcome.upper()
    text = (
        f"<b>{_PREFIX} — CLOSED {result}</b>\n"
        f"Session: {_html_escape(session)} · {direction.upper()}\n"
        f"PnL: {pnl_pct:+.2f}% · decision #{decision_id} · exec #{execution_id}"
    )
    return await _send(text, msg_type="gemini_gold_close", exec_id=execution_id)


def _decision_id_from_execution(execution) -> Optional[int]:
    notes = getattr(execution, "notes", "") or ""
    if "decision_id=" in notes:
        try:
            return int(notes.split("decision_id=")[1].split()[0].strip("|"))
        except (ValueError, IndexError):
            pass
    meta = getattr(execution, "conditions_met", None)
    if isinstance(meta, dict):
        try:
            did = int(meta.get("gemini_gold_decision_id"))
            return did if did > 0 else None
        except (TypeError, ValueError):
            pass
    return None


async def sync_closed_trade_notifications(db, cfg: GeminiGoldRuntimeConfig) -> int:
    """Record missing outcomes and notify on newly closed demo trades."""
    from datetime import datetime, timedelta

    from app.gemini_gold_trader.outcomes import record_outcome_from_execution
    from app.strategy_models import StrategyExecution

    notifications_enabled = telegram_notifications_enabled()
    since = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    closed = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.notes.like("%gemini_gold_trader%"),
            StrategyExecution.outcome != "OPEN",
            StrategyExecution.closed_at.isnot(None),
            StrategyExecution.closed_at >= since - timedelta(days=3),
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(30)
        .all()
    )
    sent = 0
    for ex in closed:
        did = _decision_id_from_execution(ex)
        if not did:
            continue
        was_new = record_outcome_from_execution(db, did, ex)
        if not was_new:
            continue
        if not notifications_enabled:
            continue
        from app.gemini_gold_trader.models import GeminiGoldDecision

        dec = db.query(GeminiGoldDecision).filter(GeminiGoldDecision.id == did).first()
        ok = await notify_trade_closed(
            session=getattr(dec, "session", None) or "—",
            direction=ex.direction or "?",
            outcome=ex.outcome or "?",
            pnl_pct=float(ex.pnl_pct or 0),
            decision_id=did,
            execution_id=ex.id,
        )
        if ok:
            sent += 1
    return sent


_last_fallback_alert: Optional[str] = None


async def maybe_notify_fallback_klines(block_reason: str, market_data: dict) -> None:
    global _last_fallback_alert
    if not telegram_notifications_enabled():
        return
    if not block_reason or block_reason == _last_fallback_alert:
        return
    if "fallback_klines" not in block_reason and "non_ctrader" not in block_reason:
        return
    _last_fallback_alert = block_reason
    ks = market_data.get("kline_source") or "unknown"
    ps = market_data.get("price_source") or "unknown"
    await _send(
        f"{_PREFIX} <b>Data gate</b>\n"
        f"Scan blocked: <code>{_html_escape(block_reason)}</code>\n"
        f"price={_html_escape(str(ps))} kline={_html_escape(str(ks))}",
        msg_type="gemini_gold_data_gate",
    )
