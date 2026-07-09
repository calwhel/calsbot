"""Execution readiness and blocked-TAKE diagnostics for Gemini Gold."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig
from app.gemini_gold_trader.guardrails import (
    active_ctrader_account_id,
    check_can_execute,
    trading_account_configured,
)
from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldPendingOrder


def _normalize_skip_reason(reason: Optional[str]) -> str:
    raw = (reason or "not_executed").strip()
    if not raw:
        return "not_executed"
    if raw.startswith("blocked: "):
        raw = raw[len("blocked: ") :]
    if raw.startswith("blocked:"):
        raw = raw[len("blocked:") :].strip()
    if raw.startswith("fire_time:"):
        return raw[len("fire_time:") :].strip()
    return raw.split(" — ", 1)[0].strip() or "not_executed"


def skip_reason_breakdown(db, *, days: int = 14, limit: int = 12) -> List[Dict[str, Any]]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(GeminiGoldDecision.skip_reason, func.count(GeminiGoldDecision.id))
        .filter(
            GeminiGoldDecision.ts >= since,
            GeminiGoldDecision.action == "TAKE",
            GeminiGoldDecision.executed.is_(False),
        )
        .group_by(GeminiGoldDecision.skip_reason)
        .order_by(func.count(GeminiGoldDecision.id).desc())
        .limit(limit)
        .all()
    )
    out: List[Dict[str, Any]] = []
    for reason, count in rows:
        out.append(
            {
                "reason": _normalize_skip_reason(reason),
                "raw_reason": reason,
                "count": int(count or 0),
            }
        )
    return out


def pending_entry_summary(db) -> Dict[str, int]:
    now = datetime.utcnow()
    try:
        rows = (
            db.query(GeminiGoldPendingOrder.status, func.count(GeminiGoldPendingOrder.id))
            .group_by(GeminiGoldPendingOrder.status)
            .all()
        )
        summary = {str(status or "unknown"): int(count or 0) for status, count in rows}
        summary["active"] = int(
            db.query(GeminiGoldPendingOrder)
            .filter(
                GeminiGoldPendingOrder.status == "pending",
                (GeminiGoldPendingOrder.expires_at.is_(None))
                | (GeminiGoldPendingOrder.expires_at > now),
            )
            .count()
        )
        return summary
    except Exception:
        return {"active": 0}


def execution_stats(db, *, days: int = 14) -> Dict[str, Any]:
    since = datetime.utcnow() - timedelta(days=days)
    takes = (
        db.query(func.count(GeminiGoldDecision.id))
        .filter(GeminiGoldDecision.ts >= since, GeminiGoldDecision.action == "TAKE")
        .scalar()
        or 0
    )
    executed = (
        db.query(func.count(GeminiGoldDecision.id))
        .filter(
            GeminiGoldDecision.ts >= since,
            GeminiGoldDecision.action == "TAKE",
            GeminiGoldDecision.executed.is_(True),
        )
        .scalar()
        or 0
    )
    rate = round(100.0 * executed / takes, 1) if takes else 0.0
    return {
        "takes": int(takes),
        "executed": int(executed),
        "execute_rate_pct": rate,
    }


def build_execution_readiness(
    db,
    *,
    cfg: GeminiGoldRuntimeConfig,
    user_id: Optional[int],
    account_snap: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Checklist explaining why orders may not reach cTrader."""
    issues: List[str] = []
    checks: List[Dict[str, Any]] = []

    def _check(name: str, ok: bool, ok_detail: str, fail_detail: str) -> None:
        checks.append({"name": name, "ok": ok, "detail": ok_detail if ok else fail_detail})
        if not ok:
            issues.append(fail_detail)

    _check(
        "trading_account",
        trading_account_configured(cfg),
        f"Account #{active_ctrader_account_id(cfg) or '?'} configured",
        "Demo/live cTrader account not configured",
    )
    _check(
        "demo_user",
        bool(cfg.demo_user_id),
        f"Trader user #{cfg.demo_user_id} linked",
        "Trader user not linked (GEMINI_GOLD_USER_ID)",
    )
    if cfg.dry_run:
        _check("dry_run_off", False, "Dry-run OFF", "Dry-run is ON — orders blocked before broker")
    else:
        _check("dry_run_off", True, "Dry-run OFF — orders allowed to broker", "Dry-run is ON")
    _check(
        "kill_switch_off",
        not cfg.kill_switch,
        "Kill switch off",
        "Kill switch is ON",
    )

    if user_id:
        can_exec, reason = check_can_execute(db, cfg, int(user_id))
        _check(
            "caps_ok",
            can_exec,
            "Caps OK",
            f"Execution caps: {reason}",
        )
    else:
        _check("caps_ok", False, "Caps OK", "No trader user for cap check")

    broker_ok = True
    broker_ok_detail = "Broker connected"
    broker_fail_detail = "Broker poll not run"
    if account_snap:
        broker_ok = not account_snap.get("broker_unreachable")
        if account_snap.get("balance_error") == "no_ctrader_token":
            broker_fail_detail = "cTrader OAuth token missing — re-link in portal"
            broker_ok = False
        elif account_snap.get("broker_unreachable"):
            broker_fail_detail = account_snap.get("balance_error") or "Broker unreachable"
            broker_ok = False
        elif account_snap.get("balance") is not None or account_snap.get("equity") is not None:
            bal = account_snap.get("balance")
            eq = account_snap.get("equity")
            broker_ok_detail = f"Broker connected (bal={bal}, eq={eq})"
        elif account_snap.get("balance_cached"):
            broker_fail_detail = "Broker poll failed — showing cached balance"
            broker_ok = False
        else:
            broker_fail_detail = "Broker balance unavailable"
            broker_ok = False
    _check("broker_reachable", broker_ok, broker_ok_detail, broker_fail_detail)

    stats = execution_stats(db, days=14)
    blockers = skip_reason_breakdown(db, days=14, limit=12)
    if not cfg.dry_run:
        blockers = [b for b in blockers if b.get("reason") != "dry_run"]
    pending = pending_entry_summary(db)

    ready = len(issues) == 0
    primary_blocker = blockers[0]["reason"] if blockers else (issues[0] if issues else None)

    return {
        "ready": ready,
        "issues": issues,
        "checks": checks,
        "stats_14d": stats,
        "top_blockers": blockers,
        "primary_blocker": primary_blocker,
        "pending_entries": pending,
    }
