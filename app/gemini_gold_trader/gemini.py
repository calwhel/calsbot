"""Gemini 2.5 Flash vision API — structured trade decisions from chart images."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig, SYMBOL

logger = logging.getLogger(__name__)

# Gemini 2.5 Flash paid-tier pricing (per 1M tokens, Jun 2026)
_INPUT_COST_PER_M = 0.30
_OUTPUT_COST_PER_M = 2.50
_CALL_TIMEOUT_S = max(15.0, float(os.environ.get("GEMINI_GOLD_DECIDE_TIMEOUT_S", "45")))


class GeminiGoldDecisionSchema(BaseModel):
    action: str = Field(description="TAKE or SKIP")
    direction: Optional[str] = Field(
        default=None, description="LONG or SHORT when action is TAKE, else null"
    )
    entry: Optional[float] = Field(default=None, description="Planned entry price for TAKE")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss price for TAKE")
    take_profit: Optional[float] = Field(default=None, description="Take profit price for TAKE")
    confidence: int = Field(ge=0, le=100, description="Confidence 0-100")
    rationale: str = Field(description="What you see on the charts and why you chose this action")


def _get_gemini_client():
    try:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("AI_INTEGRATIONS_GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception:
        return None


def _build_prompt(*, session: str, spot: float, bars_15m: int, bars_1h: int) -> str:
    return (
        "You are an experienced XAUUSD SCALPER — you hunt fast, session-local setups "
        "that can play out in minutes to a few hours. You are NOT a swing trader. "
        "You do not target multi-day moves, distant daily/weekly levels, or slow "
        "HTF trendline plays that need hours to develop. You are decisive and "
        "risk-first: skip unclear chop rather than force a trade.\n\n"
        "SCALP MANDATE (read this before every decision):\n"
        "- PRIMARY trigger timeframe: 15-minute chart — entry, sweep, OB reaction, "
        "and MSS must be visible and actionable NOW on 15m.\n"
        "- 1-hour chart: context/bias ONLY (session trend, nearby structure). Never "
        "TAKE solely because 1h \"looks bullish/bearish\" without a live 15m scalp trigger.\n"
        "- Hold horizon: same session, quick exit at TP — not overnight, not \"wait for "
        "HTF target\".\n"
        "- SKIP any setup that needs a wide stop, a distant entry, or a multi-hour "
        "developing pattern — that is swing trading, not scalping.\n\n"
        "SCALP RISK RULES (mandatory for every TAKE):\n"
        "- Stop loss: 30–150 platform pips from entry (XAUUSD: 1 pip = $0.10 → "
        "30 pips ≈ $3, 150 pips ≈ $15 max).\n"
        "- Take profit: risk:reward 1:1 to 2:1 only — no 3R+ swing targets.\n"
        "- Entry near current price — no \"limit at distant OB\" scalps; if price isn't "
        "there now with a 15m trigger, SKIP.\n"
        "- If the setup cannot fit 30–150 pip SL with 1–2R TP at current price, SKIP.\n\n"
        f"Charts: 15m ({bars_15m} candles) = PRIMARY scalp trigger; "
        f"1h ({bars_1h} candles) = bias/context only.\n"
        f"Session: {session}. Spot reference: {spot:.2f}.\n\n"
        "PRICE ANCHORING (mandatory):\n"
        "- Chart prices and spot reference are ground truth.\n"
        "- Base entry, stop_loss, take_profit on visible chart levels only.\n\n"
        "SCALP SETUP VOCABULARY — TAKE only when you can name ONE of these with "
        "15m price levels and an immediate trigger (not \"maybe later\"):\n\n"
        "1. SESSION LIQUIDITY SWEEP + RECLAIM (best scalp) — PDH/PDL, Asian range "
        "high/low, EQH/EQL, or prior session swing swept and reclaimed with 15m "
        "displacement (e.g. \"sweep below PDL 2638, reclaimed, bullish 15m MSS\").\n\n"
        "2. OPENING RANGE BREAKOUT (ORB) — current session ORB break + hold on 15m; "
        "cite range and break level.\n\n"
        "3. 15m ORDER BLOCK / FVG REACTION — price AT or just into a fresh OB or FVG "
        "on 15m with rejection/displacement now (not a distant 1h zone price hasn't "
        "reached).\n\n"
        "4. 15m MOMENTUM SCALP — flag/consolidation break or EMA pullback bounce in "
        "session direction on 15m; cite the flag/EMA zone.\n\n"
        "5. LIQUIDITY GRAB + 15m MSS — quick sweep of a nearby pool then structure "
        "break on 15m in reversal direction.\n\n"
        "SWING SETUPS TO SKIP (do not TAKE even if tempting):\n"
        "- Multi-touch 1h trendline plays needing hours to resolve.\n"
        "- Targeting far HTF levels (prior week high, major daily S/R) as TP.\n"
        "- \"Uptrend on 1h, wait for pullback\" without a live 15m entry trigger.\n"
        "- Stops beyond 150 pips or TPs implying 3R+ / multi-session holds.\n"
        "- Generic trend narrative without a named scalp pattern above.\n\n"
        "DECISION RULES:\n"
        "- TAKE: named scalp pattern + 15m trigger + 30–150 pip SL + 1–2R TP + "
        "confidence reflects real conviction.\n"
        "- SKIP: no qualifying scalp, swing-style setup, or chop — say which scalp "
        "patterns you checked and why none qualify.\n\n"
        "If TAKE: direction, entry near spot, stop_loss, take_profit (1–2R, 30–150 pip SL), "
        "confidence 0–100, rationale naming the scalp pattern and 15m levels.\n\n"
        "If SKIP: action SKIP, null prices, confidence 0–100, rationale plain and specific."
    )


def _estimate_cost(tokens_in: int, tokens_out: int) -> float:
    return round(
        (tokens_in / 1_000_000.0) * _INPUT_COST_PER_M
        + (tokens_out / 1_000_000.0) * _OUTPUT_COST_PER_M,
        6,
    )


def _parse_usage(response) -> Tuple[int, int]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return 0, 0
    tin = int(getattr(usage, "prompt_token_count", 0) or 0)
    tout = int(getattr(usage, "candidates_token_count", 0) or 0)
    if tout == 0:
        tout = int(getattr(usage, "total_token_count", 0) or 0) - tin
    return max(0, tin), max(0, tout)


def _normalize_decision(raw: Dict[str, Any]) -> Dict[str, Any]:
    action = str(raw.get("action") or "SKIP").strip().upper()
    if action not in ("TAKE", "SKIP"):
        action = "SKIP"
    direction = raw.get("direction")
    if direction is not None:
        direction = str(direction).strip().upper()
        if direction not in ("LONG", "SHORT"):
            direction = None
    try:
        confidence = int(raw.get("confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0
    confidence = max(0, min(100, confidence))

    def _f(key: str) -> Optional[float]:
        v = raw.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    return {
        "action": action,
        "direction": direction,
        "entry": _f("entry"),
        "stop_loss": _f("stop_loss"),
        "take_profit": _f("take_profit"),
        "confidence": confidence,
        "rationale": str(raw.get("rationale") or "").strip(),
        "symbol": SYMBOL,
    }


async def decide_from_charts(
    *,
    cfg: GeminiGoldRuntimeConfig,
    session: str,
    spot: float,
    png_15m: bytes,
    png_1h: bytes,
    bars_15m: int,
    bars_1h: int,
) -> Tuple[Optional[Dict[str, Any]], int, int, float, Optional[str]]:
    """
    Call Gemini vision. Returns (decision_dict, tokens_in, tokens_out, cost_usd, error).
    """
    client = _get_gemini_client()
    if not client:
        return None, 0, 0, 0.0, "no_gemini_api_key"

    from google.genai import types as genai_types

    prompt = _build_prompt(session=session, spot=spot, bars_15m=bars_15m, bars_1h=bars_1h)

    def _call():
        return client.models.generate_content(
            model=cfg.model,
            contents=[
                genai_types.Part.from_bytes(data=png_15m, mime_type="image/png"),
                genai_types.Part.from_bytes(data=png_1h, mime_type="image/png"),
                prompt,
            ],
            config=genai_types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=512,
                response_mime_type="application/json",
                response_schema=GeminiGoldDecisionSchema,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )

    try:
        response = await asyncio.wait_for(asyncio.to_thread(_call), timeout=_CALL_TIMEOUT_S)
    except asyncio.TimeoutError:
        return None, 0, 0, 0.0, "gemini_timeout"
    except Exception as exc:
        logger.warning("[gemini-gold] API error: %s", exc)
        return None, 0, 0, 0.0, f"gemini_error:{exc}"

    tokens_in, tokens_out = _parse_usage(response)
    cost = _estimate_cost(tokens_in, tokens_out)

    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            raw = parsed.model_dump()
        elif isinstance(parsed, dict):
            raw = parsed
        else:
            raw = {}
    else:
        import json

        text = (getattr(response, "text", None) or "").strip()
        if not text:
            return None, tokens_in, tokens_out, cost, "empty_response"
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            return None, tokens_in, tokens_out, cost, "invalid_json"

    return _normalize_decision(raw), tokens_in, tokens_out, cost, None
