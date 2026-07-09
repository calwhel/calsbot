"""AI performance review — Gemini Pro analyzes trades, funnel, and cTrader account."""
from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig
from app.gemini_gold_trader.funnel import snapshot as funnel_snapshot
from app.gemini_gold_trader.funnel_persist import recent_funnel_events
from app.gemini_gold_trader.guardrails import (
    active_ctrader_account_id,
    calls_today,
    cost_today_usd,
    trades_today,
)
from app.gemini_gold_trader.learning import call_stats_today, get_setup_stats
from app.gemini_gold_trader.models import GeminiGoldDecision, GeminiGoldOutcome, GeminiGoldReview

logger = logging.getLogger(__name__)

# Pro-tier pricing (Jun 2026) — review uses stronger model than scan flash
_REVIEW_INPUT_COST_PER_M = 1.25
_REVIEW_OUTPUT_COST_PER_M = 10.00
_REVIEW_TIMEOUT_S = max(30.0, float(os.environ.get("GEMINI_GOLD_REVIEW_TIMEOUT_S", "90")))
_CTRADER_REVIEW_TIMEOUT_S = max(5.0, float(os.environ.get("GEMINI_GOLD_REVIEW_CTRADER_TIMEOUT_S", "12")))

APPLYABLE_CONFIG_FIELDS = frozenset(
    {
        "confidence_threshold",
        "max_trades_day",
        "max_calls_day",
        "demo_lot_size",
        "live_lot_size",
        "use_limit_entry",
        "orb_enabled",
        "orb_confidence_threshold",
        "orb_max_calls_day",
        "orb_max_trades_per_session",
        "max_live_trades_day",
    }
)


class GeminiGoldReviewSuggestionSchema(BaseModel):
    field: str = Field(description="Config field name e.g. confidence_threshold")
    current_value: Optional[str] = Field(default=None)
    suggested_value: str = Field(description="New value as string")
    reason: str = Field(description="Why change this — cite trade data")


class GeminiGoldReviewResultSchema(BaseModel):
    summary: str = Field(description="2–4 sentence executive summary")
    whats_working: List[str] = Field(description="Patterns and setups that are profitable")
    whats_not_working: List[str] = Field(description="Failure patterns with specifics")
    setup_insights: List[str] = Field(description="Per-setup-type notes with win rate references")
    timing_insights: List[str] = Field(
        default_factory=list,
        description="Hour-of-day, session timing, hold duration, and time-between-trade patterns",
    )
    aggressiveness_insights: List[str] = Field(
        default_factory=list,
        description="Trade frequency vs caps, confidence distribution, lot sizing, take/execute rates",
    )
    ctrader_account_notes: str = Field(
        default="",
        description="cTrader broker account health: balance/equity, open positions, recent closes, reconciliation",
    )
    funnel_diagnosis: str = Field(description="Where the pipeline loses edge before execution")
    lesson_for_next_sessions: str = Field(
        description="Short rules Gemini should follow on future scans (4–8 bullets in one string)"
    )
    config_suggestions: List[GeminiGoldReviewSuggestionSchema] = Field(
        default_factory=list,
        description="Concrete portal config tweaks — only applyable fields",
    )


def review_model_name() -> str:
    return (
        os.environ.get("GEMINI_GOLD_REVIEW_MODEL", "gemini-2.5-pro").strip()
        or "gemini-2.5-pro"
    )


def _estimate_review_cost(tokens_in: int, tokens_out: int) -> float:
    return round(
        (tokens_in / 1_000_000.0) * _REVIEW_INPUT_COST_PER_M
        + (tokens_out / 1_000_000.0) * _REVIEW_OUTPUT_COST_PER_M,
        6,
    )


def get_latest_review_lesson(db) -> Optional[str]:
    """Latest review lesson injected into live scan prompts."""
    row = (
        db.query(GeminiGoldReview)
        .order_by(GeminiGoldReview.ts.desc())
        .first()
    )
    if not row:
        return None
    lesson = str(row.lesson_for_next_sessions or "").strip()
    return lesson or None


def recent_reviews(db, *, limit: int = 5) -> List[Dict[str, Any]]:
    rows = (
        db.query(GeminiGoldReview)
        .order_by(GeminiGoldReview.ts.desc())
        .limit(max(1, min(limit, 20)))
        .all()
    )
    out = []
    for row in rows:
        out.append(
            {
                "id": row.id,
                "ts": row.ts.isoformat() if row.ts else None,
                "summary": row.summary,
                "whats_working": row.whats_working or [],
                "whats_not_working": row.whats_not_working or [],
                "setup_insights": row.setup_insights or [],
                "timing_insights": getattr(row, "timing_insights", None) or [],
                "aggressiveness_insights": getattr(row, "aggressiveness_insights", None) or [],
                "ctrader_account_notes": getattr(row, "ctrader_account_notes", None),
                "funnel_diagnosis": row.funnel_diagnosis,
                "lesson_for_next_sessions": row.lesson_for_next_sessions,
                "config_suggestions": row.config_suggestions or [],
                "model": row.model,
                "cost_usd": float(row.cost_usd or 0),
                "days_window": row.days_window,
            }
        )
    return out


def filter_applyable_changes(changes: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, val in (changes or {}).items():
        if key not in APPLYABLE_CONFIG_FIELDS:
            continue
        if val is None:
            continue
        out[key] = val
    return out


def suggestions_to_changes(suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
    changes: Dict[str, Any] = {}
    for s in suggestions or []:
        field = str(s.get("field") or "").strip()
        if field not in APPLYABLE_CONFIG_FIELDS:
            continue
        raw = s.get("suggested_value")
        if raw is None:
            continue
        if field in (
            "confidence_threshold",
            "max_trades_day",
            "max_calls_day",
            "orb_confidence_threshold",
            "orb_max_calls_day",
            "orb_max_trades_per_session",
            "max_live_trades_day",
            "pending_entry_timeout_min",
        ):
            try:
                changes[field] = int(raw)
            except (TypeError, ValueError):
                continue
        elif field in ("demo_lot_size", "live_lot_size"):
            try:
                changes[field] = float(raw)
            except (TypeError, ValueError):
                continue
        elif field in ("use_limit_entry", "orb_enabled"):
            changes[field] = str(raw).strip().lower() in ("1", "true", "yes", "on")
        else:
            changes[field] = raw
    return changes


def _recent_broker_closes_block(db, *, user_id: int, days: int, limit: int = 15) -> List[Dict[str, Any]]:
    """Recent gemini gold closes from StrategyExecution (cTrader ground truth)."""
    from app.strategy_models import StrategyExecution

    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(StrategyExecution)
        .filter(
            StrategyExecution.user_id == user_id,
            StrategyExecution.symbol == "XAUUSD",
            StrategyExecution.notes.like("%gemini_gold_trader%"),
            StrategyExecution.closed_at.isnot(None),
            StrategyExecution.closed_at >= since,
            StrategyExecution.outcome.in_(("WIN", "LOSS", "BREAKEVEN", "CANCELLED")),
        )
        .order_by(StrategyExecution.closed_at.desc())
        .limit(limit)
        .all()
    )
    out: List[Dict[str, Any]] = []
    for ex in rows:
        hold_min = None
        if ex.fired_at and ex.closed_at:
            hold_min = round((ex.closed_at - ex.fired_at).total_seconds() / 60.0, 1)
        out.append(
            {
                "execution_id": ex.id,
                "fired_at": ex.fired_at.isoformat() if ex.fired_at else None,
                "closed_at": ex.closed_at.isoformat() if ex.closed_at else None,
                "direction": ex.direction,
                "outcome": ex.outcome,
                "entry_price": float(ex.entry_price) if ex.entry_price else None,
                "exit_price": float(ex.exit_price) if ex.exit_price else None,
                "pnl_pct": float(ex.pnl_pct) if ex.pnl_pct is not None else None,
                "pnl_usd": float(ex.pnl_usd) if ex.pnl_usd is not None else None,
                "hold_min": hold_min,
                "ctrader_account_id": ex.ctrader_account_id,
                "broker_position_id": ex.ctrader_position_id,
            }
        )
    return out


async def _fetch_ctrader_account_snapshot(
    db,
    *,
    user_id: int,
    cfg: GeminiGoldRuntimeConfig,
    days: int = 14,
) -> Dict[str, Any]:
    """Best-effort cTrader account equity, broker positions, and recent closes."""
    from app.gemini_gold_trader.reconcile import list_open_executions

    ctid = active_ctrader_account_id(cfg)
    snap: Dict[str, Any] = {
        "ctrader_account_id": ctid,
        "execution_mode": cfg.execution_mode,
        "balance": None,
        "equity": None,
        "balance_error": None,
        "broker_open_position_count": None,
        "broker_unreachable": False,
        "position_reconciliation": "unknown",
        "broker_open_position_ids": [],
    }
    open_execs = list_open_executions(db, user_id, demo_ctid=ctid) if user_id else []
    snap["tracked_open_positions"] = open_execs
    snap["open_positions"] = open_execs
    snap["open_position_count"] = len(open_execs)
    snap["recent_broker_closes"] = _recent_broker_closes_block(db, user_id=user_id, days=days) if user_id else []

    if user_id and snap["recent_broker_closes"]:
        pnl_usd = sum(float(r.get("pnl_usd") or 0) for r in snap["recent_broker_closes"])
        snap["recent_broker_pnl_usd"] = round(pnl_usd, 2)

    if not user_id or not ctid:
        return snap

    try:
        from app.models import User, UserPreference
        from app.services.ctrader_client import (
            get_account_reconcile_resilient,
            get_open_position_ids_for_user_with_retry,
        )

        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            snap["balance_error"] = "user_not_found"
            return snap
        prefs = db.query(UserPreference).filter(UserPreference.user_id == int(user_id)).first()
        token = getattr(prefs, "ctrader_access_token", None) or getattr(user, "ctrader_access_token", None)
        if not token:
            snap["balance_error"] = "no_ctrader_token"
            return snap

        acct = await asyncio.wait_for(
            get_account_reconcile_resilient(
                str(token),
                int(ctid),
                prefs=prefs,
                user_id=int(user_id),
            ),
            timeout=_CTRADER_REVIEW_TIMEOUT_S,
        )
        if acct:
            snap["balance"] = acct.get("balance")
            snap["equity"] = acct.get("equity")
        else:
            snap["balance_error"] = "balance_unavailable"

        broker_ids = await asyncio.wait_for(
            get_open_position_ids_for_user_with_retry(
                user,
                ctid=ctid,
                attempts=1,
            ),
            timeout=_CTRADER_REVIEW_TIMEOUT_S,
        )
        if broker_ids is None:
            snap["broker_unreachable"] = True
            snap["position_reconciliation"] = "unknown"
        else:
            snap["broker_open_position_count"] = len(broker_ids)
            snap["broker_open_position_ids"] = sorted(int(x) for x in broker_ids)[:20]
            tracked_ids = set()
            for pos in open_execs:
                raw = pos.get("broker_position_id") or pos.get("ctrader_position_id")
                if raw is not None:
                    try:
                        tracked_ids.add(int(str(raw).strip()))
                    except (TypeError, ValueError):
                        pass
            if len(broker_ids) == len(tracked_ids) and broker_ids == tracked_ids:
                snap["position_reconciliation"] = "match"
            elif len(broker_ids) == 0 and len(tracked_ids) == 0:
                snap["position_reconciliation"] = "flat"
            else:
                snap["position_reconciliation"] = "mismatch"
                snap["broker_only_positions"] = sorted(broker_ids - tracked_ids)
                snap["tracked_only_positions"] = sorted(tracked_ids - broker_ids)
    except asyncio.TimeoutError:
        snap["balance_error"] = "ctrader_poll_timeout"
    except Exception as exc:
        snap["balance_error"] = str(exc)[:120]
    return snap


def _timing_analysis_block(db, *, days: int) -> List[str]:
    """Hour/session timing, hold duration, and inter-trade gaps."""
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(GeminiGoldOutcome, GeminiGoldDecision)
        .join(GeminiGoldDecision, GeminiGoldDecision.id == GeminiGoldOutcome.decision_id)
        .filter(
            GeminiGoldOutcome.closed_ts.isnot(None),
            GeminiGoldOutcome.closed_ts >= since,
            GeminiGoldDecision.executed.is_(True),
        )
        .order_by(GeminiGoldDecision.ts.asc())
        .all()
    )
    lines = [f"=== TIMING ANALYSIS ({days}d) ==="]
    if not rows:
        lines.append("No executed closed trades for timing stats.")
        return lines

    hour_buckets: Dict[int, Dict[str, int]] = {}
    session_buckets: Dict[str, Dict[str, int]] = {}
    hold_mins: List[float] = []
    gaps_min: List[float] = []
    last_exec_ts: Optional[datetime] = None

    for out, dec in rows:
        ts = dec.ts or out.closed_ts
        if ts:
            h = int(ts.hour)
            b = hour_buckets.setdefault(h, {"trades": 0, "wins": 0, "losses": 0})
            b["trades"] += 1
            if out.result == "win":
                b["wins"] += 1
            elif out.result == "loss":
                b["losses"] += 1
        sess = (out.session or dec.session or "unknown").lower()
        sb = session_buckets.setdefault(sess, {"trades": 0, "wins": 0, "losses": 0})
        sb["trades"] += 1
        if out.result == "win":
            sb["wins"] += 1
        elif out.result == "loss":
            sb["losses"] += 1
        if dec.ts and out.closed_ts:
            hold_mins.append((out.closed_ts - dec.ts).total_seconds() / 60.0)
        if dec.ts and last_exec_ts:
            gaps_min.append((dec.ts - last_exec_ts).total_seconds() / 60.0)
        if dec.ts:
            last_exec_ts = dec.ts

    lines.append("Win rate by UTC hour (executed closes):")
    for h in sorted(hour_buckets):
        b = hour_buckets[h]
        wr = round(100.0 * b["wins"] / b["trades"], 1) if b["trades"] else 0
        lines.append(f"  {h:02d}:00 UTC — {b['trades']} trades, WR {wr}%")

    lines.append("Win rate by session:")
    for sess in sorted(session_buckets):
        b = session_buckets[sess]
        wr = round(100.0 * b["wins"] / b["trades"], 1) if b["trades"] else 0
        lines.append(f"  {sess}: {b['trades']} trades, WR {wr}%")

    if hold_mins:
        hold_mins.sort()
        med = hold_mins[len(hold_mins) // 2]
        lines.append(
            f"Hold time (entry→close): median {med:.0f}m, "
            f"min {hold_mins[0]:.0f}m, max {hold_mins[-1]:.0f}m, n={len(hold_mins)}"
        )
    if gaps_min:
        gaps_min.sort()
        med_gap = gaps_min[len(gaps_min) // 2]
        lines.append(
            f"Gap between executed trades: median {med_gap:.0f}m, "
            f"min {gaps_min[0]:.0f}m, max {gaps_min[-1]:.0f}m"
        )
    return lines


def _aggressiveness_block(db, *, cfg: GeminiGoldRuntimeConfig, days: int) -> List[str]:
    """Trade frequency, confidence, caps usage, and block patterns."""
    since = datetime.utcnow() - timedelta(days=days)
    decisions = (
        db.query(GeminiGoldDecision)
        .filter(GeminiGoldDecision.ts >= since)
        .order_by(GeminiGoldDecision.ts.asc())
        .all()
    )
    lines = [f"=== AGGRESSIVENESS ({days}d) ==="]
    if not decisions:
        lines.append("No Gemini decisions in window.")
        return lines

    calls = len(decisions)
    takes = sum(1 for d in decisions if (d.action or "").upper() == "TAKE")
    executed = sum(1 for d in decisions if d.executed)
    skips = sum(1 for d in decisions if (d.action or "").upper() == "SKIP")
    min_gap_blocks = sum(1 for d in decisions if (d.skip_reason or "") == "min_trade_gap")
    validator_blocks = sum(
        1 for d in decisions
        if d.action == "TAKE" and not d.executed and "validator" in (d.skip_reason or "")
    )

    conf_buckets = {"<70": 0, "70-79": 0, "80-89": 0, "90+": 0}
    conf_wins: List[int] = []
    conf_losses: List[int] = []
    trades_by_day: Dict[str, int] = {}

    for d in decisions:
        c = int(d.confidence or 0)
        if c < 70:
            conf_buckets["<70"] += 1
        elif c < 80:
            conf_buckets["70-79"] += 1
        elif c < 90:
            conf_buckets["80-89"] += 1
        else:
            conf_buckets["90+"] += 1
        if d.executed and d.ts:
            day = d.ts.date().isoformat()
            trades_by_day[day] = trades_by_day.get(day, 0) + 1

    closed = (
        db.query(GeminiGoldOutcome, GeminiGoldDecision)
        .join(GeminiGoldDecision, GeminiGoldDecision.id == GeminiGoldOutcome.decision_id)
        .filter(GeminiGoldOutcome.closed_ts.isnot(None), GeminiGoldOutcome.closed_ts >= since)
        .all()
    )
    for out, dec in closed:
        c = int(dec.confidence or 0)
        if out.result == "win":
            conf_wins.append(c)
        elif out.result == "loss":
            conf_losses.append(c)

    lines.append(f"min_trade_gap_min={cfg.min_trade_gap_min}")
    lines.append(f"max_trades_day={cfg.max_trades_day} max_calls_day={cfg.max_calls_day}")
    lines.append(f"demo_lot_size={cfg.demo_lot_size} live_lot_size={cfg.live_lot_size}")
    lines.append(
        f"Pipeline: calls={calls} takes={takes} executed={executed} skips={skips} "
        f"take_rate={round(100*takes/calls,1) if calls else 0}% "
        f"execute_rate={round(100*executed/takes,1) if takes else 0}%"
    )
    lines.append(f"Blocks: min_trade_gap={min_gap_blocks} validator={validator_blocks}")
    lines.append(
        "Confidence distribution (all decisions): "
        + ", ".join(f"{k}={v}" for k, v in conf_buckets.items())
    )
    if conf_wins:
        lines.append(f"Avg confidence on wins: {round(sum(conf_wins)/len(conf_wins), 1)}% (n={len(conf_wins)})")
    if conf_losses:
        lines.append(f"Avg confidence on losses: {round(sum(conf_losses)/len(conf_losses), 1)}% (n={len(conf_losses)})")
    if trades_by_day:
        avg_day = sum(trades_by_day.values()) / len(trades_by_day)
        peak = max(trades_by_day.values())
        lines.append(
            f"Executed trades/day: avg {avg_day:.1f}, peak {peak}, "
            f"cap {cfg.max_trades_day} ({round(100*peak/cfg.max_trades_day,0) if cfg.max_trades_day else 0}% of cap on busiest day)"
        )
    return lines


def _ctrader_account_block(account_snap: Dict[str, Any]) -> List[str]:
    lines = ["=== CTRADER BROKER ACCOUNT (live poll) ==="]
    lines.append(f"account_id={account_snap.get('ctrader_account_id')}")
    lines.append(f"execution_mode={account_snap.get('execution_mode')}")
    lines.append(f"balance_usd={account_snap.get('balance')}")
    lines.append(f"equity_usd={account_snap.get('equity')}")
    if account_snap.get("balance_error"):
        lines.append(f"account_fetch_note={account_snap.get('balance_error')}")
    lines.append(f"broker_open_positions={account_snap.get('broker_open_position_count')}")
    lines.append(f"tracked_open_positions={account_snap.get('open_position_count')}")
    lines.append(f"position_reconciliation={account_snap.get('position_reconciliation')}")
    if account_snap.get("broker_unreachable"):
        lines.append("broker_status=unreachable (could not poll open positions)")
    broker_ids = account_snap.get("broker_open_position_ids") or []
    if broker_ids:
        lines.append(f"broker_position_ids={broker_ids[:10]}")
    if account_snap.get("broker_only_positions"):
        lines.append(f"broker_only (not tracked)={account_snap.get('broker_only_positions')}")
    if account_snap.get("tracked_only_positions"):
        lines.append(f"tracked_only (phantom?)={account_snap.get('tracked_only_positions')}")

    for pos in (account_snap.get("tracked_open_positions") or [])[:5]:
        lines.append(
            f"  tracked open: {pos.get('direction')} entry={pos.get('entry_price')} "
            f"pos_id={pos.get('broker_position_id')} fired={pos.get('fired_at')}"
        )

    closes = account_snap.get("recent_broker_closes") or []
    if closes:
        lines.append(f"recent_broker_closes ({len(closes)} rows, StrategyExecution):")
        if account_snap.get("recent_broker_pnl_usd") is not None:
            lines.append(f"  sum_pnl_usd={account_snap.get('recent_broker_pnl_usd')}")
        for c in closes[:8]:
            lines.append(
                f"  - {c.get('closed_at')} | {c.get('direction')} {c.get('outcome')} | "
                f"pnl_usd={c.get('pnl_usd')} pnl_pct={c.get('pnl_pct')} | hold={c.get('hold_min')}m | "
                f"pos={c.get('broker_position_id')}"
            )
    else:
        lines.append("recent_broker_closes=none in window")
    return lines


def _recent_closed_trades_block(db, *, days: int, limit: int = 25) -> List[str]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(GeminiGoldOutcome, GeminiGoldDecision)
        .join(GeminiGoldDecision, GeminiGoldDecision.id == GeminiGoldOutcome.decision_id)
        .filter(GeminiGoldOutcome.closed_ts.isnot(None), GeminiGoldOutcome.closed_ts >= since)
        .order_by(GeminiGoldOutcome.closed_ts.desc())
        .limit(limit)
        .all()
    )
    lines = [f"=== CLOSED TRADES ({days}d, up to {limit}) ==="]
    if not rows:
        lines.append("No closed trades in window.")
        return lines
    for out, dec in rows:
        d = dec.decision if isinstance(dec.decision, dict) else {}
        lines.append(
            f"- {out.closed_ts} | {out.setup_type} | {out.session} | {out.result} | "
            f"pnl={out.pnl:+.2f}% R={out.r_multiple} | {d.get('direction')} | "
            f"entry={d.get('entry')} sl={d.get('stop_loss')} tp={d.get('take_profit')} | "
            f"conf={dec.confidence}% | executed={dec.executed} | "
            f"rationale={(dec.rationale or d.get('rationale') or '')[:100]}"
        )
    return lines


def _blocked_takes_block(db, *, days: int, limit: int = 15) -> List[str]:
    since = datetime.utcnow() - timedelta(days=days)
    rows = (
        db.query(GeminiGoldDecision)
        .filter(
            GeminiGoldDecision.ts >= since,
            GeminiGoldDecision.action == "TAKE",
            GeminiGoldDecision.executed.is_(False),
        )
        .order_by(GeminiGoldDecision.ts.desc())
        .limit(limit)
        .all()
    )
    lines = [f"=== BLOCKED TAKES ({days}d) ==="]
    if not rows:
        lines.append("No blocked TAKE decisions.")
        return lines
    for row in rows:
        lines.append(
            f"- {row.ts} | {row.setup_type} | conf={row.confidence}% | "
            f"skip/block={row.skip_reason or 'not_executed'} | "
            f"rationale={(row.rationale or '')[:90]}"
        )
    return lines


def build_review_prompt(
    db,
    *,
    cfg: GeminiGoldRuntimeConfig,
    user_id: int,
    days: int,
    account_snap: Dict[str, Any],
) -> str:
    funnel = funnel_snapshot()
    stats = get_setup_stats(db, days=days)
    calls_today_stats = call_stats_today(db)
    events = recent_funnel_events(db, limit=40)

    lines = [
        "You are reviewing the Gemini Gold XAUUSD vision trader (demo + optional live mirror).",
        f"Analysis window: last {days} days.",
        "",
        "=== CURRENT CONFIG ===",
        f"scan_model={cfg.model} (flash — used for live charts; do NOT suggest changing unless critical)",
        f"confidence_threshold={cfg.confidence_threshold}%",
        f"max_trades_day={cfg.max_trades_day}",
        f"max_calls_day={cfg.max_calls_day}",
        f"demo_lot_size={cfg.demo_lot_size}",
        f"live_lot_size={cfg.live_lot_size}",
        f"use_limit_entry={cfg.use_limit_entry}",
        f"orb_enabled={cfg.orb_enabled}",
        f"orb_confidence_threshold={cfg.orb_confidence_threshold}",
        f"min_trade_gap_min={cfg.min_trade_gap_min}",
        f"dry_run={cfg.dry_run}",
        f"execution_mode={cfg.execution_mode}",
        "",
        "=== TODAY ===",
        f"gemini_calls_today={calls_today(db)}",
        f"trades_executed_today={trades_today(db)}",
        f"api_cost_today_usd={cost_today_usd(db):.4f}",
        "",
    ]
    lines.extend(_ctrader_account_block(account_snap))
    lines.append("")
    lines.append("=== SETUP STATS ===")
    for s in stats[:12]:
        avg_r = s.get("avg_r")
        avg_r_s = f"{avg_r:.2f}" if avg_r is not None else "n/a"
        lines.append(
            f"- {s['setup_type']} ({s['session']}): {s['trades']} trades, "
            f"WR {s['win_rate']}%, avg R {avg_r_s}, pnl {s['total_pnl']:+.2f}%"
        )

    lines.append("")
    lines.append("=== CALL STATS TODAY (per setup) ===")
    for c in calls_today_stats[:12]:
        lines.append(
            f"- {c['setup_type']}: calls={c['calls']} takes={c['takes']} executed={c['executed']}"
        )

    lines.append("")
    lines.append("=== FUNNEL TODAY ===")
    for key in (
        "scans",
        "gemini_called",
        "gemini_take",
        "gemini_skip",
        "validator_rejected",
        "stale_entry_blocked",
        "executed",
        "data_blocked",
        "chart_failed",
    ):
        lines.append(f"- {key}: {funnel.get(key, 0)}")
    if funnel.get("last_validator_reason"):
        lines.append(f"- last_validator_reason: {funnel.get('last_validator_reason')}")

    lines.append("")
    lines.append("=== RECENT FUNNEL EVENTS ===")
    for ev in events[:20]:
        lines.append(
            f"- {ev.get('ts')} | {ev.get('event')} | setup={ev.get('setup_type')} | "
            f"reason={ev.get('reason')}"
        )

    lines.extend(_recent_closed_trades_block(db, days=days))
    lines.extend(_blocked_takes_block(db, days=days))
    lines.extend(_timing_analysis_block(db, days=days))
    lines.extend(_aggressiveness_block(db, cfg=cfg, days=days))

    lines.append("")
    lines.append(
        "Produce a structured review. You MUST analyze timing (hours/sessions/hold times) "
        "and aggressiveness (frequency vs caps, confidence, min_trade_gap). "
        "ctrader_account_notes must summarize the cTrader broker account section "
        "(balance/equity, open positions, reconciliation, recent closes). "
        "config_suggestions.field must be one of: "
        + ", ".join(sorted(APPLYABLE_CONFIG_FIELDS))
        + ". Only suggest changes backed by the trade data above."
    )
    return "\n".join(lines)


async def run_performance_review(
    db,
    *,
    cfg: GeminiGoldRuntimeConfig,
    user_id: int,
    days: int = 14,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Run Gemini Pro performance review. Returns (result_dict, error).
    Persists GeminiGoldReview row on success.
    """
    from app.gemini_gold_trader.gemini import _get_gemini_client, _parse_usage

    client = _get_gemini_client()
    if not client:
        return None, "no_gemini_api_key"

    account_snap = await _fetch_ctrader_account_snapshot(db, user_id=user_id, cfg=cfg, days=days)
    prompt = build_review_prompt(db, cfg=cfg, user_id=user_id, days=days, account_snap=account_snap)
    model = review_model_name()

    from google.genai import types as genai_types

    system = (
        "You are a senior XAUUSD trading systems analyst reviewing an AI vision scalper.\n"
        "Be specific — cite setup types, sessions, win rates, blocked TAKE patterns, "
        "timing (UTC hours, hold duration), and aggressiveness (trade frequency vs caps).\n"
        "Always comment on the cTrader broker account: balance/equity, open positions, "
        "tracked vs broker reconciliation, and recent broker closes.\n"
        "config_suggestions must use exact field names and realistic values for gold scalping.\n"
        "lesson_for_next_sessions will be injected into every future Gemini scan prompt — make it actionable."
    )

    def _call():
        return client.models.generate_content(
            model=model,
            contents=f"{system}\n\n{prompt}",
            config=genai_types.GenerateContentConfig(
                temperature=0.25,
                max_output_tokens=4096,
                response_mime_type="application/json",
                response_schema=GeminiGoldReviewResultSchema,
            ),
        )

    try:
        response = await asyncio.wait_for(asyncio.to_thread(_call), timeout=_REVIEW_TIMEOUT_S)
    except asyncio.TimeoutError:
        return None, "review_timeout"
    except Exception as exc:
        logger.warning("[gemini-gold] review API error: %s", exc)
        return None, f"review_error:{exc}"

    tokens_in, tokens_out = _parse_usage(response)
    cost = _estimate_review_cost(tokens_in, tokens_out)

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        raw = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    else:
        import json

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            return None, "empty_response"
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return None, "invalid_json"

    suggestions = []
    for s in raw.get("config_suggestions") or []:
        if isinstance(s, dict):
            suggestions.append(s)

    row = GeminiGoldReview(
        summary=str(raw.get("summary") or "").strip(),
        whats_working=list(raw.get("whats_working") or []),
        whats_not_working=list(raw.get("whats_not_working") or []),
        setup_insights=list(raw.get("setup_insights") or []),
        timing_insights=list(raw.get("timing_insights") or []),
        aggressiveness_insights=list(raw.get("aggressiveness_insights") or []),
        ctrader_account_notes=str(raw.get("ctrader_account_notes") or "").strip(),
        funnel_diagnosis=str(raw.get("funnel_diagnosis") or "").strip(),
        lesson_for_next_sessions=str(raw.get("lesson_for_next_sessions") or "").strip(),
        config_suggestions=suggestions,
        model=model,
        days_window=int(days),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        account_snapshot=account_snap,
    )
    try:
        db.add(row)
        db.commit()
        db.refresh(row)
    except Exception as exc:
        db.rollback()
        logger.exception("[gemini-gold] review persist failed")
        return None, f"review_persist:{exc}"

    result = {
        "id": row.id,
        "ts": row.ts.isoformat() if row.ts else None,
        "summary": row.summary,
        "whats_working": row.whats_working,
        "whats_not_working": row.whats_not_working,
        "setup_insights": row.setup_insights,
        "timing_insights": row.timing_insights,
        "aggressiveness_insights": row.aggressiveness_insights,
        "ctrader_account_notes": row.ctrader_account_notes,
        "funnel_diagnosis": row.funnel_diagnosis,
        "lesson_for_next_sessions": row.lesson_for_next_sessions,
        "config_suggestions": row.config_suggestions,
        "model": row.model,
        "cost_usd": cost,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "days_window": days,
        "applyable_changes": suggestions_to_changes(suggestions),
    }
    logger.info("[gemini-gold] performance review id=%s model=%s cost=$%.4f", row.id, model, cost)
    return result, None
