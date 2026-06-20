"""Haiku screen → Opus confirm funnel (shadow mode default)."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

from app.gold_ai_trader.claude import _parse_json, decide
from app.gold_ai_trader.config import (
    HAIKU_CACHE_READ_USD_PER_M,
    HAIKU_CACHE_WRITE_USD_PER_M,
    HAIKU_INPUT_USD_PER_M,
    HAIKU_OUTPUT_USD_PER_M,
    OPUS_INPUT_USD_PER_M,
    OPUS_OUTPUT_USD_PER_M,
    SCREEN_MODEL,
)
from app.gold_ai_trader.context import build_screen_context
from app.gold_ai_trader.models import GoldAiFunnelFalseReject
from app.gold_ai_trader.scanner import Candidate

logger = logging.getLogger(__name__)

SCREEN_SYSTEM_PROMPT = """You are a cheap pre-screener for XAUUSD gold setups. You are NOT the decision-maker — Claude Opus confirms every trade.

OVER-PASS RULE (critical): When unsure → PASS. False rejects lose edge. False passes only cost one Opus call.

SKIP only when the setup is clearly non-viable:
1. Dead tape — RVOL below floor, holiday/no participation, no activity
2. Price obviously extended far past any entry zone (mechanical distance in context, not judgment)
3. No coherent trigger present at all

If ANY part of the trigger could be valid → PASS. Borderline → PASS. Marginal → PASS.

Output ONLY valid JSON (no prose):
{"screen": "PASS" | "SKIP", "reason": "<8 words max>"}"""


def funnel_mode_from_env() -> str:
    raw = (os.environ.get("GOLD_FUNNEL_MODE") or "shadow").strip().lower()
    return raw if raw in ("shadow", "live") else "shadow"


def maybe_geometry_prefilter(
    candidate: Candidate,
    price: float,
    atr: float,
    *,
    rvol: float = 1.0,
) -> Tuple[bool, Optional[str]]:
    """Insertion point for future mechanical RVOL/zone/ATR gate before Haiku."""
    _ = (candidate, price, atr, rvol)
    return True, None


def _estimate_haiku_cost(usage) -> Tuple[int, int, int, int, float]:
    tin = int(getattr(usage, "input_tokens", 0) or 0)
    tout = int(getattr(usage, "output_tokens", 0) or 0)
    cr = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cw = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    billable_in = max(0, tin - cr)
    cost = (
        billable_in / 1_000_000 * HAIKU_INPUT_USD_PER_M
        + tout / 1_000_000 * HAIKU_OUTPUT_USD_PER_M
        + cr / 1_000_000 * HAIKU_CACHE_READ_USD_PER_M
        + cw / 1_000_000 * HAIKU_CACHE_WRITE_USD_PER_M
    )
    return tin, tout, cr, cw, round(cost, 6)


def is_false_reject(screen_action: str, opus_action: str, opus_confidence: int) -> bool:
    """Haiku SKIP but Opus would have taken or rated high-confidence."""
    if (screen_action or "").upper() != "SKIP":
        return False
    if (opus_action or "").lower() == "take":
        return True
    return int(opus_confidence or 0) >= 50


async def screen_setup(
    context_text: str,
    *,
    model: str = SCREEN_MODEL,
    dry_run: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Cheap Haiku screen — returns ({screen, reason}, usage_meta)."""
    if dry_run:
        return (
            {"screen": "PASS", "reason": "dry-run over-pass"},
            {
                "tokens_in": 0,
                "tokens_out": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "cost_usd": 0.0,
                "cache_control_applied": True,
            },
        )

    api_key = (
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("AI_INTEGRATIONS_ANTHROPIC_API_KEY")
    )
    if not api_key:
        logger.warning("[gold-ai-trader/funnel] no Anthropic API key — screen defaults PASS")
        return (
            {"screen": "PASS", "reason": "no api key"},
            {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "cache_control_applied": False},
        )

    try:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model=model,
            max_tokens=40,
            system=[
                {
                    "type": "text",
                    "text": SCREEN_SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Screen this gold setup. JSON only.\n\n" + context_text
                    ),
                }
            ],
        )
        text = (msg.content[0].text or "").strip() if msg.content else ""
        tin, tout, cr, cw, cost = _estimate_haiku_cost(msg.usage)
        meta = {
            "tokens_in": tin,
            "tokens_out": tout,
            "cache_read_tokens": cr,
            "cache_write_tokens": cw,
            "cost_usd": cost,
            "cache_control_applied": True,
        }
        parsed = _parse_json(text) or {}
        screen = str(parsed.get("screen", "PASS")).upper()
        if screen not in ("PASS", "SKIP"):
            screen = "PASS"
        reason = str(parsed.get("reason", "") or "")[:80]
        return {"screen": screen, "reason": reason or "parse fallback pass"}, meta
    except Exception as exc:
        logger.warning("[gold-ai-trader/funnel] Haiku screen failed: %s — default PASS", exc)
        return (
            {"screen": "PASS", "reason": "screen error pass"},
            {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "cache_control_applied": False},
        )


@dataclass
class FunnelResult:
    funnel_mode: str
    screen_action: str
    screen_reason: str
    screen_usage: Dict[str, Any]
    opus_called: bool
    decision: Dict[str, Any]
    reasoning: str
    opus_usage: Dict[str, Any]
    false_reject: bool
    geometry_skipped: bool


async def run_funnel_pipeline(
    *,
    candidate: Candidate,
    price: float,
    atr: float,
    rvol: float,
    session: str,
    full_context: str,
    cfg,
    db,
    dry_run: bool = False,
) -> FunnelResult:
    """Run geometry hook → Haiku screen → Opus confirm (shadow always calls Opus)."""
    funnel_mode = getattr(cfg, "funnel_mode", "shadow") or "shadow"
    geo_ok, geo_reason = maybe_geometry_prefilter(candidate, float(price), float(atr), rvol=rvol)
    if not geo_ok:
        return FunnelResult(
            funnel_mode=funnel_mode,
            screen_action="SKIP",
            screen_reason=geo_reason or "geometry prefilter",
            screen_usage={"cost_usd": 0.0},
            opus_called=False,
            decision={
                "action": "skip",
                "confidence": 0,
                "rationale": geo_reason or "geometry prefilter",
            },
            reasoning=geo_reason or "geometry prefilter",
            opus_usage={"cost_usd": 0.0},
            false_reject=False,
            geometry_skipped=True,
        )

    screen_ctx = await build_screen_context(
        candidate=candidate,
        price=price,
        session=session,
        atr=atr,
        rvol=rvol,
    )
    screen_model = getattr(cfg, "screen_model", None) or SCREEN_MODEL
    screen_out, screen_usage = await screen_setup(screen_ctx, model=screen_model, dry_run=dry_run)
    screen_action = str(screen_out.get("screen", "PASS")).upper()
    screen_reason = str(screen_out.get("reason", "") or "")

    short_circuit = funnel_mode == "live" and screen_action == "SKIP"
    opus_called = not short_circuit

    if short_circuit:
        decision = {
            "action": "skip",
            "direction": None,
            "entry": None,
            "stop_loss": None,
            "take_profit": None,
            "confidence": 0,
            "rationale": f"Haiku screen SKIP: {screen_reason}",
        }
        reasoning = decision["rationale"]
        opus_usage: Dict[str, Any] = {
            "tokens_in": 0,
            "tokens_out": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "cost_usd": 0.0,
        }
    else:
        decision, reasoning, opus_usage = await decide(full_context, model=cfg.model, dry_run=dry_run)

    action = (decision.get("action") or "skip").lower()
    conf = int(decision.get("confidence") or 0)
    false_reject = funnel_mode == "shadow" and is_false_reject(screen_action, action, conf)

    screen_cost = float(screen_usage.get("cost_usd", 0) or 0)
    opus_cost = float(opus_usage.get("cost_usd", 0) or 0)
    total_cost = round(screen_cost + opus_cost, 6)

    logger.info(
        "[gold-ai-trader/funnel] mode=%s screen=%s reason=%r opus_called=%s "
        "opus_action=%s conf=%s screen_cost=$%.4f opus_cost=$%.4f total=$%.4f "
        "false_reject=%s cache_screen=%s",
        funnel_mode,
        screen_action,
        screen_reason,
        opus_called,
        action,
        conf,
        screen_cost,
        opus_cost,
        total_cost,
        false_reject,
        screen_usage.get("cache_control_applied"),
    )

    return FunnelResult(
        funnel_mode=funnel_mode,
        screen_action=screen_action,
        screen_reason=screen_reason,
        screen_usage=screen_usage,
        opus_called=opus_called,
        decision=decision,
        reasoning=reasoning,
        opus_usage=opus_usage,
        false_reject=false_reject,
        geometry_skipped=False,
    )


def persist_false_reject(
    db,
    *,
    decision_id: int,
    session: str,
    candidate_type: str,
    screen_reason: str,
    opus_action: str,
    opus_confidence: int,
    opus_rationale: str,
) -> None:
    row = GoldAiFunnelFalseReject(
        decision_id=decision_id,
        session=session,
        candidate_type=candidate_type,
        screen_reason=screen_reason,
        opus_action=opus_action,
        opus_confidence=int(opus_confidence or 0),
        opus_rationale=(opus_rationale or "")[:500],
    )
    db.add(row)
    db.commit()


def _calls_cutoff(db):
    from app.gold_ai_trader.guardrails import _calls_cutoff

    return _calls_cutoff(db)


def funnel_stats_today(db) -> Dict[str, Any]:
    """Daily funnel instrumentation for logs and status API."""
    from sqlalchemy import func

    from app.gold_ai_trader.models import GoldAiDecision

    cutoff = _calls_cutoff(db)
    rows = (
        db.query(GoldAiDecision)
        .filter(GoldAiDecision.ts >= cutoff)
        .all()
    )
    screened = sum(1 for r in rows if getattr(r, "screen_action", None))
    passed = sum(1 for r in rows if (getattr(r, "screen_action", "") or "").upper() == "PASS")
    skipped = sum(1 for r in rows if (getattr(r, "screen_action", "") or "").upper() == "SKIP")
    opus_calls = sum(
        1
        for r in rows
        if getattr(r, "opus_called", True) is not False
    )
    screen_cost = sum(float(getattr(r, "screen_cost_usd", 0) or 0) for r in rows)
    opus_cost = sum(float(getattr(r, "cost_usd", 0) or 0) for r in rows)
    total_cost = sum(float(getattr(r, "total_cost_usd", 0) or getattr(r, "cost_usd", 0) or 0) for r in rows)
    false_rejects = (
        db.query(func.count(GoldAiFunnelFalseReject.id))
        .filter(GoldAiFunnelFalseReject.ts >= cutoff)
        .scalar()
        or 0
    )

    avg_opus_cost = (opus_cost / opus_calls) if opus_calls else 0.0
    hypothetical_all_opus = total_cost + (skipped * avg_opus_cost if avg_opus_cost else 0.0)
    savings_vs_all_opus = max(0.0, hypothetical_all_opus - total_cost)
    escalation_rate = (opus_calls / screened * 100.0) if screened else 0.0
    blended_cost = (total_cost / screened) if screened else 0.0

    stats = {
        "funnel_mode": funnel_mode_from_env(),
        "screened": screened,
        "screen_pass": passed,
        "screen_skip": skipped,
        "opus_called": opus_calls,
        "escalation_rate_pct": round(escalation_rate, 1),
        "false_rejects": int(false_rejects),
        "screen_cost_usd": round(screen_cost, 4),
        "opus_cost_usd": round(opus_cost, 4),
        "total_cost_usd": round(total_cost, 4),
        "blended_cost_per_decision_usd": round(blended_cost, 4),
        "hypothetical_all_opus_cost_usd": round(hypothetical_all_opus, 4),
        "projected_savings_vs_all_opus_usd": round(savings_vs_all_opus, 4),
    }
    logger.info("[gold-ai-trader/funnel] daily stats %s", stats)
    return stats
