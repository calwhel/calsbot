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


async def _fetch_ctrader_account_snapshot(
    db,
    *,
    user_id: int,
    cfg: GeminiGoldRuntimeConfig,
) -> Dict[str, Any]:
    """Best-effort cTrader account equity + open gemini positions."""
    from app.gemini_gold_trader.reconcile import list_open_executions

    ctid = active_ctrader_account_id(cfg)
    snap: Dict[str, Any] = {
        "ctrader_account_id": ctid,
        "execution_mode": cfg.execution_mode,
        "balance": None,
        "balance_error": None,
    }
    open_execs = list_open_executions(db, user_id, demo_ctid=ctid) if user_id else []
    snap["open_positions"] = open_execs
    snap["open_position_count"] = len(open_execs)

    if not user_id or not ctid:
        return snap

    try:
        from app.models import User, UserPreference
        from app.services.ctrader_client import get_account_balance_resilient

        user = db.query(User).filter(User.id == int(user_id)).first()
        if not user:
            snap["balance_error"] = "user_not_found"
            return snap
        prefs = db.query(UserPreference).filter(UserPreference.user_id == int(user_id)).first()
        token = getattr(prefs, "ctrader_access_token", None) or getattr(user, "ctrader_access_token", None)
        if not token:
            snap["balance_error"] = "no_ctrader_token"
            return snap
        bal = await get_account_balance_resilient(
            str(token),
            int(ctid),
            prefs=prefs,
            user_id=int(user_id),
        )
        if bal is not None and float(bal) > 0:
            snap["balance"] = round(float(bal), 2)
        else:
            snap["balance_error"] = "balance_unavailable"
    except Exception as exc:
        snap["balance_error"] = str(exc)[:120]
    return snap


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
        f"dry_run={cfg.dry_run}",
        f"execution_mode={cfg.execution_mode}",
        "",
        "=== TODAY ===",
        f"gemini_calls_today={calls_today(db)}",
        f"trades_executed_today={trades_today(db)}",
        f"api_cost_today_usd={cost_today_usd(db):.4f}",
        "",
        "=== CTRADER ACCOUNT ===",
        f"account_id={account_snap.get('ctrader_account_id')}",
        f"balance={account_snap.get('balance')}",
        f"balance_note={account_snap.get('balance_error')}",
        f"open_positions={account_snap.get('open_position_count')}",
    ]
    for pos in (account_snap.get("open_positions") or [])[:5]:
        lines.append(
            f"  open: {pos.get('direction')} entry={pos.get('entry_price')} "
            f"pos_id={pos.get('broker_position_id')}"
        )

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

    lines.append("")
    lines.append(
        "Produce a structured review. config_suggestions.field must be one of: "
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

    account_snap = await _fetch_ctrader_account_snapshot(db, user_id=user_id, cfg=cfg)
    prompt = build_review_prompt(db, cfg=cfg, user_id=user_id, days=days, account_snap=account_snap)
    model = review_model_name()

    from google.genai import types as genai_types

    system = (
        "You are a senior XAUUSD trading systems analyst reviewing an AI vision scalper.\n"
        "Be specific — cite setup types, sessions, win rates, and blocked TAKE patterns.\n"
        "config_suggestions must use exact field names and realistic values for gold scalping.\n"
        "lesson_for_next_sessions will be injected into every future Gemini scan prompt — make it actionable."
    )

    def _call():
        return client.models.generate_content(
            model=model,
            contents=f"{system}\n\n{prompt}",
            config=genai_types.GenerateContentConfig(
                temperature=0.25,
                max_output_tokens=2500,
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
    db.add(row)
    db.commit()
    db.refresh(row)

    result = {
        "id": row.id,
        "ts": row.ts.isoformat() if row.ts else None,
        "summary": row.summary,
        "whats_working": row.whats_working,
        "whats_not_working": row.whats_not_working,
        "setup_insights": row.setup_insights,
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
