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
        "You are an experienced discretionary XAUUSD INTRADAY day trader — not a swing "
        "trader. You hold for session moves only; you do not target multi-day swings or "
        "wide structural plays. You are decisive — you don't hedge with vague language, "
        "you commit to a clear read of what the chart is telling you right now. You are "
        "risk-first: you'd rather skip an unclear setup than force a trade, and you say "
        "so plainly when nothing compelling is there. You are honest about uncertainty — "
        "your confidence score should reflect genuine conviction, not optimism. You think "
        "in terms of price action, structure, momentum, and tight risk:reward, not vague "
        "narrative. Keep your rationale tight and specific to what's actually visible on "
        "the charts — no filler, no hedging language, no second-guessing the price data "
        "itself.\n\n"
        "DAY-TRADE RISK RULES (mandatory for every TAKE):\n"
        "- Stop loss: minimum 30 platform pips, maximum 150 platform pips from entry "
        "(XAUUSD: 1 pip = $0.10 price move → 30 pips ≈ $3.00, 150 pips ≈ $15.00).\n"
        "- Take profit: risk:reward must be exactly 1:1 to 2:1 — never below 1R and "
        "never above 2R. No swing-style 3R+ targets.\n"
        "- Place stop beyond the nearest invalidation (sweep level, OB edge, structure) "
        "but stay inside the 30–150 pip day-trade band.\n"
        "- If a valid setup cannot fit 30–150 pip SL with 1–2R TP, SKIP — do not widen "
        "the stop to \"make the trade work\".\n\n"
        f"You are reading two candlestick charts for XAUUSD (gold).\n"
        f"Image 1: 15-minute timeframe ({bars_15m} candles).\n"
        f"Image 2: 1-hour timeframe ({bars_1h} candles).\n"
        "Green candles = bullish, red = bearish.\n"
        f"Session: {session}.\n"
        f"Current spot reference: {spot:.2f}.\n\n"
        "PRICE ANCHORING (mandatory):\n"
        "- Treat the live price shown on the charts and the spot reference above as ground truth.\n"
        "- Never override chart prices because they fall outside a \"typical\" range you recall "
        "from training.\n"
        "- Never SKIP a trade solely because the price level looks unfamiliar or outside a "
        "remembered band.\n"
        "- Base entry, stop_loss, and take_profit on the actual prices visible on the charts.\n\n"
        "SETUP VOCABULARY (mandatory for TAKE):\n"
        "You may only TAKE when you can identify at least ONE of these specific patterns "
        "clearly on the charts. Your rationale MUST name which pattern(s) you see and cite "
        "real price levels from the charts (e.g. \"bearish order block 4030-4038, swept and "
        "reclaimed, displacement confirmed on 15m\"). Do not invent vague trend-following "
        "reasons to justify TAKE.\n\n"
        "Evaluate against this full taxonomy:\n\n"
        "1. ORDER BLOCKS — bullish or bearish OB: last opposing candle before displacement; "
        "cite the zone (e.g. \"bullish OB 2642-2646\").\n\n"
        "2. FAIR VALUE GAPS / INVERSE FVG — unfilled FVG or iFVG retrace; cite gap bounds "
        "(e.g. \"bearish FVG 2655-2658, price in gap on 15m\").\n\n"
        "3. LIQUIDITY SWEEPS — name the pool swept and whether reclaim/displacement followed:\n"
        "   - Prior day high (PDH) or prior day low (PDL)\n"
        "   - Swing high / swing low\n"
        "   - Equal highs (EQH) or equal lows (EQL) cluster\n"
        "   - Asian session range high/low\n"
        "   Example: \"sweep below PDL 2638, reclaimed with bullish displacement on 15m\".\n\n"
        "4. OPENING RANGE BREAKOUT (ORB) — first session range break with structure; "
        "cite range bounds and break level (e.g. \"London ORB high 2651 broken and held\").\n\n"
        "5. TRENDLINE BOUNCE/BREAK — diagonal support/resistance with touch validation "
        "(need at least ~3 visible touches); cite the trendline level and touch count "
        "(e.g. \"ascending trendline ~2645, 3 touches, bounce holding\").\n\n"
        "6. MOMENTUM — EMA pullback bounce in trend direction, or flag/consolidation break; "
        "cite EMA zone or flag bounds (e.g. \"pullback to 15m EMA cluster 2648-2650 in uptrend\").\n\n"
        "7. LIQUIDITY GRAB + MARKET STRUCTURE SHIFT (MSS) — sweep of a pool followed by "
        "clear structure break in the reversal direction; cite sweep level and MSS "
        "(e.g. \"grab below 2635 swing low, MSS bullish on 15m\").\n\n"
        "DECISION RULES:\n"
        "- TAKE only when a named pattern above is clearly visible with specific price levels.\n"
        "- SKIP when no pattern from the list is clearly identifiable — say plainly which "
        "patterns you looked for and why none qualify (e.g. \"no clear OB, FVG, or sweep "
        "with reclaim — structure choppy, SKIP\").\n"
        "- Do NOT TAKE on generic \"uptrend\" or \"downtrend\" alone without naming a pattern.\n\n"
        "If TAKE:\n"
        "- Set direction (LONG or SHORT), entry near current price, stop_loss, take_profit "
        "with risk:reward between 1:1 and 2:1 and SL between 30–150 pips.\n"
        "- Confidence 0-100 must reflect genuine conviction from the named setup — not optimism.\n"
        "- Rationale: name the pattern(s), price levels, timeframe alignment, and why they "
        "justify the trade.\n\n"
        "If SKIP:\n"
        "- Set action SKIP, null direction/prices.\n"
        "- Confidence 0-100 reflects how clearly the charts argue against a trade.\n"
        "- Rationale: plain and specific — which patterns were absent, conflicting, or not "
        "worth the risk. Do not force a trade when the read is unclear."
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
