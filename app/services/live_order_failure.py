"""Live cTrader order failure classification, durable logging, retry, and Telegram alerts."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import traceback
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_S = (0.5, 1.5)
_MAX_LIVE_ORDER_ATTEMPTS = 3


class FailureCategory(str, Enum):
    SKIPPED = "skipped"
    BROKER_REJECTED = "broker_rejected"
    EXCEPTION = "exception"


@dataclass
class ClassifiedFailure:
    reason: str
    category: FailureCategory
    transient: bool
    broker_reply: Optional[str] = None


def _norm(text: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def humanize_skip_reason(blockers: List[str]) -> str:
    """Turn readiness gate blockers into one clear string."""
    if not blockers:
        return "cTrader not ready"
    parts = []
    for b in blockers:
        low = b.lower()
        if "forex" in low and "approv" in low:
            parts.append("forex not approved")
        elif "token" in low or "credential" in low:
            parts.append("token expired or missing")
        elif "account" in low:
            parts.append("no cTrader account selected")
        else:
            parts.append(b)
    return "; ".join(dict.fromkeys(parts))


def classify_live_order_failure(
    raw_error: Optional[str],
    *,
    exception: Optional[BaseException] = None,
    broker_reply: Optional[Any] = None,
    skip_reason: Optional[str] = None,
) -> ClassifiedFailure:
    """Classify a live-order failure and whether it is safe to retry."""
    if skip_reason:
        return ClassifiedFailure(
            reason=_norm(skip_reason),
            category=FailureCategory.SKIPPED,
            transient=False,
        )

    if exception is not None:
        exc_type = type(exception).__name__
        exc_msg = _norm(str(exception) or exc_type)
        tb_tail = _norm("".join(traceback.format_exception_only(type(exception), exception)))
        reason = exc_msg if exc_msg else tb_tail or exc_type
        low = reason.lower()
        transient = any(
            tok in low
            for tok in (
                "timeout",
                "connection",
                "operationalerror",
                "pendingrollback",
                "temporary",
                "unavailable",
                "reset",
            )
        )
        return ClassifiedFailure(
            reason=reason[:240],
            category=FailureCategory.EXCEPTION,
            transient=transient,
            broker_reply=tb_tail[:500] if tb_tail else None,
        )

    err = _norm(raw_error or "order failed")
    low = err.lower()
    broker_txt = None
    if broker_reply is not None:
        try:
            broker_txt = json.dumps(broker_reply, default=str)[:500]
        except Exception:
            broker_txt = _norm(str(broker_reply))[:500]

    # Permanent broker / gate failures — never retry
    permanent_markers = (
        "insufficient",
        "not enough margin",
        "margin",
        "forex not approved",
        "not approved",
        "missing ctrader",
        "no ctrader",
        "invalid sl/tp",
        "could not resolve tradable volume",
        "below min",
        "minimum volume",
        "min volume",
        "invalid volume",
        "market closed",
        "not tradable",
        "symbol ",
        "queue full",
        "stale_guard",
        "signal stale",
        "no enabled",
        "status=paper",
        "ctrader_client_id",
    )
    if any(m in low for m in permanent_markers):
        return ClassifiedFailure(
            reason=_humanize_broker_error(err),
            category=FailureCategory.BROKER_REJECTED,
            transient=False,
            broker_reply=broker_txt,
        )

    if "order_cancelled" in low or "order_rejected" in low or "rejected" in low:
        return ClassifiedFailure(
            reason=_humanize_broker_error(err),
            category=FailureCategory.BROKER_REJECTED,
            transient=False,
            broker_reply=broker_txt,
        )

    # Transient — retry before notify
    transient_markers = (
        "timeout",
        "account auth failed",
        "token",
        "no execution event",
        "unexpected exit",
        "connection",
        "temporary",
        "unavailable",
        "ambiguous",
    )
    if any(m in low for m in transient_markers):
        return ClassifiedFailure(
            reason=_humanize_broker_error(err),
            category=FailureCategory.BROKER_REJECTED,
            transient=True,
            broker_reply=broker_txt,
        )

    return ClassifiedFailure(
        reason=_humanize_broker_error(err),
        category=FailureCategory.BROKER_REJECTED,
        transient=False,
        broker_reply=broker_txt,
    )


def _humanize_broker_error(err: str) -> str:
    """Map broker text to a clear human reason string."""
    up = err.upper()
    low = err.lower()
    if "INSUFFICIENT" in up or "NOT ENOUGH MARGIN" in up:
        return "insufficient margin"
    if "ORDER_CANCELLED" in up:
        tail = err.split(":", 1)[-1].strip()
        if tail and tail.upper() not in ("ORDER_CANCELLED", "ORDER CANCELLED"):
            return _humanize_broker_error(tail)
        return "broker ORDER_CANCELLED"
    if "ORDER_REJECTED" in up:
        tail = err.split(":", 1)[-1].strip()
        return _humanize_broker_error(tail) if tail else "broker ORDER_REJECTED"
    if "could not resolve tradable volume" in low:
        return "invalid volume for symbol (below broker minimum or bad step)"
    if "account auth failed" in low:
        return "token expired or account auth failed"
    if "missing ctrader account" in low:
        return "missing cTrader account id"
    if "no ctrader access token" in low or "no cTrader credentials" in low:
        return "token expired or missing"
    if "timeout" in low:
        return "broker timeout"
    if "forex not approved" in low:
        return "forex not approved"
    vol_m = re.search(r"volume\s+([\d.]+).*min(?:imum)?\s+([\d.]+)", low)
    if vol_m:
        return f"volume {vol_m.group(1)} below min {vol_m.group(2)}"
    return err[:240]


def live_order_retry_backoff_s(attempt_index: int) -> float:
    """Backoff before retry attempt_index (0-based, after first failure)."""
    if attempt_index < len(_RETRY_BACKOFF_S):
        return _RETRY_BACKOFF_S[attempt_index]
    return _RETRY_BACKOFF_S[-1]


def max_live_order_attempts() -> int:
    return _MAX_LIVE_ORDER_ATTEMPTS


def ensure_live_fire_failures_table(bind) -> None:
    """Create live_fire_failures if missing (idempotent)."""
    global _LIVE_FIRE_FAILURES_TABLE_READY
    if _LIVE_FIRE_FAILURES_TABLE_READY:
        return
    try:
        from app.strategy_models import LiveFireFailure
        LiveFireFailure.__table__.create(bind=bind, checkfirst=True)
        _LIVE_FIRE_FAILURES_TABLE_READY = True
        return
    except Exception as exc:
        logger.debug("[live-fire-fail] ORM create: %s", exc)
    try:
        from sqlalchemy import text
        with bind.connect() as conn:
            conn.execute(text(_CREATE_LIVE_FIRE_FAILURES_SQL))
            conn.commit()
        _LIVE_FIRE_FAILURES_TABLE_READY = True
    except Exception as exc:
        logger.warning("[live-fire-fail] ensure table failed: %s", exc)


_LIVE_FIRE_FAILURES_TABLE_READY = False

_CREATE_LIVE_FIRE_FAILURES_SQL = """
    CREATE TABLE IF NOT EXISTS live_fire_failures (
        id SERIAL PRIMARY KEY,
        ts TIMESTAMP NOT NULL DEFAULT NOW(),
        user_id INTEGER NOT NULL REFERENCES users(id),
        strategy_id INTEGER,
        execution_id INTEGER,
        signal_group_id VARCHAR(40),
        ctrader_account_id VARCHAR(40),
        symbol VARCHAR(30),
        direction VARCHAR(10),
        lots VARCHAR(20),
        reason TEXT NOT NULL,
        category VARCHAR(32) NOT NULL,
        attempts INTEGER NOT NULL DEFAULT 1,
        broker_reply TEXT,
        sibling_summary TEXT
    )
"""


def record_live_fire_failure(
    *,
    user_id: int,
    strategy_id: Optional[int],
    execution_id: Optional[int],
    signal_group_id: Optional[str],
    ctid: Optional[str],
    symbol: Optional[str],
    direction: Optional[str],
    lots: Optional[str],
    classified: ClassifiedFailure,
    attempts: int = 1,
    sibling_summary: Optional[str] = None,
) -> None:
    """Persist failure so it survives log rotation."""
    try:
        from app.database import SessionLocal
        from app.strategy_models import LiveFireFailure

        db = SessionLocal()
        try:
            ensure_live_fire_failures_table(db.get_bind())
            row = LiveFireFailure(
                user_id=user_id,
                strategy_id=strategy_id,
                execution_id=execution_id,
                signal_group_id=signal_group_id,
                ctrader_account_id=(ctid or "")[:40] or None,
                symbol=(symbol or "")[:30] or None,
                direction=(direction or "")[:10] or None,
                lots=(lots or "")[:20] or None,
                reason=classified.reason[:2000],
                category=classified.category.value,
                attempts=max(1, attempts),
                broker_reply=(classified.broker_reply or "")[:2000] or None,
                sibling_summary=(sibling_summary or "")[:500] or None,
            )
            db.add(row)
            db.commit()
            logger.warning(
                "[live-fire-fail] user=%s exec=%s ctid=%s reason=%s attempts=%s",
                user_id,
                execution_id,
                ctid,
                classified.reason,
                attempts,
            )
        except Exception as exc:
            logger.warning("[live-fire-fail] persist failed: %s", exc)
            try:
                db.rollback()
            except Exception:
                pass
        finally:
            db.close()
    except Exception as exc:
        logger.warning("[live-fire-fail] record skipped: %s", exc)


def fanout_sibling_summary(db, signal_group_id: Optional[str], current_exec_id: int) -> Optional[str]:
    """Summarize other legs in a fan-out signal for Telegram context."""
    if not signal_group_id:
        return None
    try:
        from app.strategy_models import StrategyExecution

        rows = (
            db.query(StrategyExecution)
            .filter(StrategyExecution.signal_group_id == signal_group_id)
            .order_by(StrategyExecution.id)
            .all()
        )
        if len(rows) <= 1:
            return None
        parts: List[str] = []
        for row in rows:
            ctid = row.ctrader_account_id or "?"
            label = f"#{ctid}"
            if row.id == current_exec_id:
                parts.append(f"{label} FAILED (this leg)")
                continue
            notes = row.notes or ""
            if row.ctrader_position_id or (
                row.ctrader_order_id
                and str(row.ctrader_order_id) not in ("queued", "None")
                and not row.is_paper
            ):
                parts.append(f"{label} filled")
            elif row.is_paper and "order_queued" not in notes:
                parts.append(f"{label} paper fallback")
            elif "order_queued" in notes:
                parts.append(f"{label} queued")
            else:
                parts.append(f"{label} pending")
        return ", ".join(parts)
    except Exception as exc:
        logger.debug("[live-fire-fail] sibling summary: %s", exc)
        return None


def _format_lots(job_or_dict) -> str:
    lots = getattr(job_or_dict, "fixed_lots", None)
    if lots is None and isinstance(job_or_dict, dict):
        lots = job_or_dict.get("fixed_lots") or job_or_dict.get("lot_size")
    if lots is None:
        return "default"
    try:
        return f"{float(lots):.2f}"
    except (TypeError, ValueError):
        return str(lots)


async def notify_live_order_failure(
    *,
    user,
    strategy_name: str,
    ctid: Optional[str],
    symbol: str,
    direction: str,
    lots: str,
    reason: str,
    attempts: int,
    sibling_summary: Optional[str] = None,
    asset_class: str = "forex",
    paper_fallback: bool = False,
    portal_settings=None,
) -> None:
    """Send Telegram alert when a live leg fails (especially partial fan-out)."""
    try:
        from app.services.strategy_executor import (
            _should_dm_trade_alerts,
            _telegram_int_id,
            _tg_send,
        )

        if portal_settings is not None and not _should_dm_trade_alerts(portal_settings, False):
            return
        tg_id = _telegram_int_id(user)
        if not tg_id:
            return

        acct_line = f"LIVE #{ctid}" if ctid else "LIVE (no account)"
        sib = f"\n<i>Other accounts in this signal: {sibling_summary}</i>" if sibling_summary else ""
        if paper_fallback:
            body = (
                "⚠️ <b>Live order fell back to paper</b>\n"
                f"Strategy: <b>{strategy_name}</b>\n"
                f"Account: {acct_line}\n"
                f"Symbol: {symbol} {direction} {lots} lots\n"
                f"Reason: <code>{reason}</code>\n"
                f"Attempts: {attempts}"
                f"{sib}"
            )
        else:
            body = (
                "⚠️ <b>LIVE ORDER DID NOT FIRE</b>\n"
                f"Strategy: <b>{strategy_name}</b>\n"
                f"Account: {acct_line}\n"
                f"Symbol: {symbol} {direction} {lots} lots\n"
                f"Reason: <code>{reason}</code>\n"
                f"Attempts: {attempts}"
                f"{sib}"
            )
        asyncio.create_task(_tg_send(tg_id, body, asset_class=asset_class))
    except Exception as exc:
        logger.warning("[live-fire-fail] telegram notify failed: %s", exc)


def list_recent_live_fire_failures(user_id: int, limit: int = 20) -> List[Dict[str, Any]]:
    """Recent durable failures for Live Forex UI."""
    try:
        from app.database import SessionLocal
        from app.strategy_models import LiveFireFailure

        db = SessionLocal()
        try:
            ensure_live_fire_failures_table(db.get_bind())
            rows = (
                db.query(LiveFireFailure)
                .filter(LiveFireFailure.user_id == user_id)
                .order_by(LiveFireFailure.ts.desc())
                .limit(max(1, min(limit, 50)))
                .all()
            )
            return [
                {
                    "ts": r.ts.isoformat() if r.ts else None,
                    "strategy_id": r.strategy_id,
                    "execution_id": r.execution_id,
                    "ctrader_account_id": r.ctrader_account_id,
                    "ctid": r.ctrader_account_id,
                    "symbol": r.symbol,
                    "direction": r.direction,
                    "lots": r.lots,
                    "reason": r.reason,
                    "category": r.category,
                    "attempts": r.attempts,
                    "sibling_summary": r.sibling_summary,
                }
                for r in rows
            ]
        finally:
            db.close()
    except Exception as exc:
        logger.debug("[live-fire-fail] list recent: %s", exc)
        return []
