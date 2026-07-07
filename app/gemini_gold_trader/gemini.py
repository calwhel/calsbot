"""Gemini 2.5 Flash vision API — structured trade decisions from chart images."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from app.gemini_gold_trader.chart_renderer import summarize_bars_for_prompt
from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig, SYMBOL
from app.gemini_gold_trader.setup_types import normalize_setup_type, setup_vocabulary_prompt_block

logger = logging.getLogger(__name__)

# Gemini 2.5 Flash paid-tier pricing (per 1M tokens, Jun 2026)
_INPUT_COST_PER_M = 0.30
_OUTPUT_COST_PER_M = 2.50
_CALL_TIMEOUT_S = max(15.0, float(os.environ.get("GEMINI_GOLD_DECIDE_TIMEOUT_S", "45")))


class GeminiGoldDecisionSchema(BaseModel):
    action: str = Field(description="TAKE or SKIP")
    setup_type: Optional[str] = Field(
        default=None,
        description=(
            "Scalp pattern id e.g. fvg_retrace_bull, ifvg_bear, ob_bull, liq_sweep_bull, "
            "liquidity_grab_long, momentum_flag_break_long, orb_long"
        ),
    )
    direction: Optional[str] = Field(
        default=None, description="LONG or SHORT when action is TAKE, else null"
    )
    entry: Optional[float] = Field(default=None, description="Planned entry price for TAKE")
    stop_loss: Optional[float] = Field(default=None, description="Stop loss price for TAKE")
    take_profit: Optional[float] = Field(default=None, description="Take profit price for TAKE")
    orb_break_level: Optional[float] = Field(default=None, description="ORB break level when setup is ORB")
    orb_range_height: Optional[float] = Field(default=None, description="ORB range height when setup is ORB")
    liq_grab_mss_level: Optional[float] = Field(default=None, description="Liquidity grab MSS level")
    momentum_break_level: Optional[float] = Field(default=None, description="Momentum break level")
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


def _build_prompt(
    *,
    session: str,
    spot: float,
    bars_1m: int,
    bars_5m: int,
    bars_15m: int,
    bars_1h: int,
    entry_timeframe: str = "1m",
    entry_5m_fallback: bool = False,
    min_sl_pips: float = 10.0,
    max_sl_pips: float = 150.0,
) -> str:
    entry_label = "1-minute" if entry_timeframe == "1m" else "5-minute (1m unavailable)"
    entry_note = (
        "- Image 1 is recent 5m bars (broker 1m unavailable) — use it for entry "
        "timing only; still require a live 5m trigger on Image 2.\n"
        if entry_5m_fallback
        else "- 1-minute chart: refine entry timing and confirm 5m trigger is live "
        "(micro rejection, engulfing, break of 1m structure) — never TAKE on 1m "
        "alone without a aligned 5m scalp trigger.\n"
    )
    min_sl = max(1.0, float(min_sl_pips))
    max_sl = max(min_sl, float(max_sl_pips))
    sl_range = f"{min_sl:.0f}–{max_sl:.0f}"
    return (
        "You are an experienced XAUUSD SCALPER — you hunt fast, session-local setups "
        "that can play out in minutes to a few hours. You are NOT a swing trader. "
        "You do not target multi-day moves, distant daily/weekly levels, or slow "
        "HTF trendline plays that need hours to develop. You are decisive and "
        "risk-first: skip unclear chop rather than force a trade.\n\n"
        "SCALP MANDATE (read this before every decision):\n"
        "- You receive FOUR charts in order (low → high timeframe). Each chart is a "
        "white-background candlestick image: green candles = bullish, red = bearish, "
        "with price on the Y-axis and bar index on the X-axis.\n"
        f"  Image 1: {entry_label} ({bars_1m} candles) — entry timing / micro confirmation.\n"
        f"  Image 2: 5-minute ({bars_5m} candles) — PRIMARY scalp trigger.\n"
        f"  Image 3: 15-minute ({bars_15m} candles) — structure, zones, session levels.\n"
        f"  Image 4: 1-hour ({bars_1h} candles) — bias/context ONLY.\n"
        "- PRIMARY trigger timeframe: 5-minute chart — sweep, OB/FVG reaction, ORB "
        "break, momentum, and MSS must be visible and actionable NOW on 5m.\n"
        f"{entry_note}"
        "- 15-minute chart: structure and zone context — where OBs/FVGs/session "
        "levels sit; must align with 5m trigger, not replace it.\n"
        "- 1-hour chart: session bias ONLY. Never TAKE solely because 1h \"looks "
        "bullish/bearish\" without a live 5m scalp trigger.\n"
        "- Hold horizon: same session, quick exit at TP — not overnight, not \"wait "
        "for HTF target\".\n"
        "- SKIP any setup that needs a wide stop, a distant entry, or a multi-hour "
        "developing pattern — that is swing trading, not scalping.\n\n"
        "SCALP RISK RULES (mandatory for every TAKE):\n"
        f"- Stop loss: {sl_range} platform pips from entry (XAUUSD: 1 pip = $0.10 → "
        f"{min_sl:.0f} pips ≈ ${min_sl * 0.1:.1f}, {max_sl:.0f} pips ≈ ${max_sl * 0.1:.1f} max).\n"
        "- Take profit: risk:reward 1:1 to 2:1 only — no 3R+ swing targets.\n"
        "- Entry near current price — no \"limit at distant OB\" scalps; if price isn't "
        "there now with a 5m trigger, SKIP.\n"
        f"- If the setup cannot fit {sl_range} pip SL with 1–2R TP at current price, SKIP.\n\n"
        f"Session: {session}. Spot reference: {spot:.2f}.\n\n"
        "PRICE ANCHORING (mandatory):\n"
        "- Chart prices and spot reference are ground truth.\n"
        "- Base entry, stop_loss, take_profit on visible chart levels only.\n\n"
        "SCALP SETUP VOCABULARY — TAKE only when you can name ONE of these with "
        "5m price levels and an immediate trigger (not \"maybe later\"):\n\n"
        "1. SESSION LIQUIDITY SWEEP + RECLAIM (best scalp) — PDH/PDL, Asian range "
        "high/low, EQH/EQL, or prior session swing swept and reclaimed with 5m "
        "displacement (e.g. \"sweep below PDL 2638, reclaimed, bullish 5m MSS\").\n\n"
        "2. OPENING RANGE BREAKOUT (ORB) — current session ORB break + hold on 5m; "
        "cite range and break level.\n\n"
        "3. 5m ORDER BLOCK / FVG / IFVG REACTION — price AT or just into a fresh OB, "
        "FVG, or inverted FVG (IFVG) on 5m (ideally inside a 15m zone) with "
        "rejection/displacement now. Shaded zones on charts mark engine-detected "
        "FVG (green/red), IFVG (blue/orange), and OB zones.\n\n"
        "4. 5m MOMENTUM SCALP — flag/consolidation break or EMA pullback bounce in "
        "session direction on 5m; cite the flag/EMA zone.\n\n"
        "5. LIQUIDITY GRAB + 5m MSS — quick sweep of a nearby pool then structure "
        "break on 5m in reversal direction; use 1m for entry timing if visible.\n\n"
        "SWING SETUPS TO SKIP (do not TAKE even if tempting):\n"
        "- Multi-touch 1h trendline plays needing hours to resolve.\n"
        "- Targeting far HTF levels (prior week high, major daily S/R) as TP.\n"
        "- \"Uptrend on 1h, wait for pullback\" without a live 5m entry trigger.\n"
        "- 15m-only setups with no live 5m trigger (too slow for a scalp).\n"
        "- Stops beyond 150 pips or TPs implying 3R+ / multi-session holds.\n"
        "- Generic trend narrative without a named scalp pattern above.\n\n"
        "DECISION RULES:\n"
        f"- TAKE: named scalp pattern + 5m trigger (+ 1m entry confirmation when "
        f"visible) + {sl_range} pip SL + 1–2R TP + confidence reflects real conviction.\n"
        f"- {setup_vocabulary_prompt_block()}\n"
        "- SKIP: no qualifying scalp, swing-style setup, or chop — say which scalp "
        "patterns you checked on 1m/5m/15m and why none qualify.\n\n"
        f"If TAKE: setup_type (exact id), direction, entry near spot, stop_loss, "
        f"take_profit (1–2R, {sl_range} pip SL), "
        "confidence 0–100, rationale naming the scalp pattern and 5m/15m levels.\n\n"
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
    setup_raw = str(raw.get("setup_type") or "").strip().lower() or None
    setup_type = normalize_setup_type(setup_raw, direction)
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
        "setup_type": setup_type,
        "direction": direction,
        "entry": _f("entry"),
        "stop_loss": _f("stop_loss"),
        "take_profit": _f("take_profit"),
        "orb_break_level": _f("orb_break_level"),
        "orb_range_height": _f("orb_range_height"),
        "liq_grab_mss_level": _f("liq_grab_mss_level"),
        "momentum_break_level": _f("momentum_break_level"),
        "confidence": confidence,
        "rationale": str(raw.get("rationale") or "").strip(),
        "symbol": SYMBOL,
    }


def _build_ohlc_context(
    *,
    entry_bars: list,
    bars_5m: list,
    bars_15m: list,
    bars_1h: list,
    entry_timeframe: str,
    zone_summary: Optional[str] = None,
) -> str:
    entry_label = "1m" if entry_timeframe == "1m" else "5m-entry"
    parts = [
        "NUMERIC OHLC SUMMARY (cross-check against the chart images):",
        summarize_bars_for_prompt(entry_bars, label=f"Entry {entry_label}"),
        summarize_bars_for_prompt(bars_5m, label="5m"),
        summarize_bars_for_prompt(bars_15m, label="15m"),
        summarize_bars_for_prompt(bars_1h, label="1h"),
    ]
    if zone_summary:
        parts.append(zone_summary)
    return "\n".join(parts)


def _chart_contents(
    genai_types,
    *,
    png_entry: bytes,
    png_5m: bytes,
    png_15m: bytes,
    png_1h: bytes,
    entry_timeframe: str,
    entry_5m_fallback: bool,
    bars_1m: int,
    bars_5m: int,
    bars_15m: int,
    bars_1h: int,
    prompt: str,
):
    entry_label = "5m entry (1m unavailable)" if entry_5m_fallback else "1m entry"
    labels = [
        (f"CHART 1 — {entry_label} timing ({bars_1m} candles):", png_entry),
        (f"CHART 2 — 5-minute scalp trigger ({bars_5m} candles):", png_5m),
        (f"CHART 3 — 15-minute structure ({bars_15m} candles):", png_15m),
        (f"CHART 4 — 1-hour session bias ({bars_1h} candles):", png_1h),
    ]
    parts = []
    for text, png in labels:
        parts.append(text)
        parts.append(genai_types.Part.from_bytes(data=png, mime_type="image/png"))
    parts.append(prompt)
    return parts


async def decide_from_charts(
    *,
    cfg: GeminiGoldRuntimeConfig,
    session: str,
    spot: float,
    png_1m: bytes,
    png_5m: bytes,
    png_15m: bytes,
    png_1h: bytes,
    entry_bars: list,
    bars_5m: list,
    bars_15m: list,
    bars_1h: list,
    entry_timeframe: str = "1m",
    entry_5m_fallback: bool = False,
    zone_summary: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], int, int, float, Optional[str]]:
    """
    Call Gemini vision. Returns (decision_dict, tokens_in, tokens_out, cost_usd, error).
    """
    client = _get_gemini_client()
    if not client:
        return None, 0, 0, 0.0, "no_gemini_api_key"

    from google.genai import types as genai_types

    prompt = _build_prompt(
        session=session,
        spot=spot,
        bars_1m=len(entry_bars),
        bars_5m=len(bars_5m),
        bars_15m=len(bars_15m),
        bars_1h=len(bars_1h),
        entry_timeframe=entry_timeframe,
        entry_5m_fallback=entry_5m_fallback,
        min_sl_pips=cfg.min_sl_pips,
        max_sl_pips=cfg.max_sl_pips,
    )
    ohlc_block = _build_ohlc_context(
        entry_bars=entry_bars,
        bars_5m=bars_5m,
        bars_15m=bars_15m,
        bars_1h=bars_1h,
        entry_timeframe=entry_timeframe,
        zone_summary=zone_summary,
    )
    full_prompt = f"{prompt}\n\n{ohlc_block}"

    def _call():
        return client.models.generate_content(
            model=cfg.model,
            contents=_chart_contents(
                genai_types,
                png_entry=png_1m,
                png_5m=png_5m,
                png_15m=png_15m,
                png_1h=png_1h,
                entry_timeframe=entry_timeframe,
                entry_5m_fallback=entry_5m_fallback,
                bars_1m=len(entry_bars),
                bars_5m=len(bars_5m),
                bars_15m=len(bars_15m),
                bars_1h=len(bars_1h),
                prompt=full_prompt,
            ),
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


async def decide_orb_text(
    context_text: str,
    *,
    cfg: GeminiGoldRuntimeConfig,
    confidence_threshold: int = 65,
) -> Tuple[Optional[Dict[str, Any]], int, int, float, Optional[str]]:
    """Text-only Gemini call for ORB confirmation (no charts)."""
    client = _get_gemini_client()
    if not client:
        return None, 0, 0, 0.0, "no_gemini_api_key"

    from google.genai import types as genai_types

    prompt = (
        "You are an XAUUSD ORB scalper. Evaluate ONLY the opening range breakout context below.\n"
        f"Minimum confidence to TAKE: {confidence_threshold}%.\n"
        "Return JSON with action TAKE or SKIP, setup_type orb_long or orb_short, direction, "
        "entry, stop_loss, take_profit, orb_break_level, orb_range_height, confidence, rationale.\n\n"
        f"{context_text}"
    )

    def _call():
        return client.models.generate_content(
            model=cfg.model,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=400,
                response_mime_type="application/json",
                response_schema=GeminiGoldDecisionSchema,
            ),
        )

    try:
        response = await asyncio.wait_for(asyncio.to_thread(_call), timeout=_CALL_TIMEOUT_S)
    except asyncio.TimeoutError:
        return None, 0, 0, 0.0, "gemini_timeout"
    except Exception as exc:
        return None, 0, 0, 0.0, f"gemini_error:{exc}"

    tokens_in, tokens_out = _parse_usage(response)
    cost = _estimate_cost(tokens_in, tokens_out)
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        raw = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
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
