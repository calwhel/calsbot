"""Gemini 2.5 Flash vision API — structured trade decisions from chart images."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

from app.gemini_gold_trader.confidence_scoring import (
    calibrate_confidence,
    confidence_scoring_prompt_block,
    format_confluence_block,
)
from app.gemini_gold_trader.chart_renderer import summarize_bars_for_prompt
from app.gemini_gold_trader.config import GeminiGoldRuntimeConfig, SYMBOL
from app.gemini_gold_trader.setup_types import (
    is_approved_setup_type,
    normalize_setup_type,
    setup_vocabulary_prompt_block,
)

logger = logging.getLogger(__name__)

# Gemini 2.5 Flash paid-tier pricing (per 1M tokens, Jun 2026)
_INPUT_COST_PER_M = 0.30
_OUTPUT_COST_PER_M = 2.50
_CALL_TIMEOUT_S = max(15.0, float(os.environ.get("GEMINI_GOLD_DECIDE_TIMEOUT_S", "45")))
_OBSERVE_TIMEOUT_S = max(20.0, float(os.environ.get("GEMINI_GOLD_OBSERVE_TIMEOUT_S", "50")))


class GeminiGoldChartObservationSchema(BaseModel):
    entry_chart: str = Field(
        description="1m/entry timing: micro structure, last candles, entry-timing cues"
    )
    chart_5m: str = Field(
        description="5m primary: sweeps, FVG/OB/IFVG, momentum, ORB, MSS with price levels"
    )
    chart_15m: str = Field(
        description="15m structure: zones, session levels, premium/discount alignment"
    )
    chart_1h: str = Field(description="1h session bias only — trend/context, not an entry trigger")
    key_levels: str = Field(
        description="Named levels with prices: PDH/PDL, Asian H/L, EQH/EQL, recent swings"
    )
    market_state: str = Field(
        description="Chop vs trend, session character, what matters right now at spot"
    )
    session_extension: str = Field(
        description=(
            "Where spot sits in today's visible range: extended HIGH/LOW/mid, "
            "% from range low, distance to session high/low — fade-short or fade-long zone?"
        )
    )
    early_opportunity: str = Field(
        description=(
            "EARLY scalp forming NOW: pumped-and-fade-short, dump-and-bounce-long, "
            "or fresh momentum — first 5m trigger live or not, with prices"
        )
    )
    setups_checked: str = Field(
        description="Each scalp pattern family checked: live trigger yes/no and brief why"
    )


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


def _early_opportunity_guidance() -> str:
    return (
        "EARLY OPPORTUNITY & SESSION FADE (priority XAUUSD scalps):\n"
        "- Gold often pumps or dumps hard intraday, then snaps back — hunt these EARLY.\n"
        "- EXTENDED UP (upper ~25% of session range, near day/session high, premium): "
        "best scalp is SHORT on the FIRST 5m bearish shift — liq grab above highs, "
        "bearish MSS, rejection wick, bear displacement. Do NOT wait for a full retrace "
        "or perfect OB; enter when reversal STARTS.\n"
        "- EXTENDED DOWN (lower ~25% of range, near day/session low, discount): "
        "mirror for EARLY LONG on first 5m bullish shift.\n"
        "- TP on fades: session mid, Asian mid, nearest 5m/15m structure — quick 1–2R, "
        "not a distant HTF target.\n"
        "- Fade setup_type: liquidity_grab_short/long, liq_sweep_bear/bull, "
        "eqh_sweep_bear, eql_sweep_bull, sdp_bear/sdp_bull, disp_bear/disp_bull.\n"
        "- Proactive beats perfect: clear extension + live 5m flip = TAKE at 78–88%. "
        "Exceptional fades with reclaim + displacement = 86–94%.\n"
    )


def summarize_session_extension(
    bars_5m: list,
    bars_15m: list,
    spot: float,
) -> str:
    """Numeric session-range position for prompt context (not a trade gate)."""
    highs: list[float] = []
    lows: list[float] = []
    for bars in (bars_5m, bars_15m):
        for bar in bars:
            if len(bar) < 5:
                continue
            try:
                h = float(bar[2])
                l = float(bar[3])
            except (TypeError, ValueError):
                continue
            if h > 0 and l > 0:
                highs.append(h)
                lows.append(l)
    if not highs or not lows or spot <= 0:
        return ""
    session_high = max(highs)
    session_low = min(lows)
    rng = session_high - session_low
    if rng <= 0:
        return ""
    pct_from_low = (spot - session_low) / rng * 100.0
    below_high = session_high - spot
    above_low = spot - session_low
    if pct_from_low >= 75.0:
        zone = "EXTENDED HIGH — fade-short zone (pumped, look for early bearish 5m trigger)"
    elif pct_from_low <= 25.0:
        zone = "EXTENDED LOW — fade-long zone (dumped, look for early bullish 5m trigger)"
    else:
        zone = "MID-RANGE — prefer sweep/ORB/momentum; fades need clearer extension"
    return (
        "SESSION RANGE POSITION (visible 5m+15m bars):\n"
        f"- Range high {session_high:.2f} | low {session_low:.2f} | spot {spot:.2f}\n"
        f"- Spot is {pct_from_low:.0f}% up from range low "
        f"(${above_low:.2f} above low, ${below_high:.2f} below high)\n"
        f"- {zone}"
    )


def observation_blocks_decide(obs: Optional[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    Step-1 gate: skip expensive decide call when observation says no live setup.
    Conservative — only blocks explicit 'no live trigger' phrasing.
    """
    if not obs:
        return False, ""
    sc = str(obs.get("setups_checked") or "").lower()
    early = str(obs.get("early_opportunity") or "").lower()
    combined = f"{sc} {early}"
    block_phrases = (
        "no live trigger",
        "none qualify",
        "no qualifying scalp",
        "no clear setup",
        "nothing live",
        "no early opportunity",
    )
    if any(p in combined for p in block_phrases):
        return True, "observation_no_live_setup"
    return False, ""


def format_chart_observation(obs: Dict[str, Any]) -> str:
    """Format step-1 observation dict as text for step-2 decision prompt."""
    if not obs:
        return ""
    sections = [
        ("Entry / timing", obs.get("entry_chart")),
        ("5-minute (primary)", obs.get("chart_5m")),
        ("15-minute structure", obs.get("chart_15m")),
        ("1-hour bias", obs.get("chart_1h")),
        ("Key levels", obs.get("key_levels")),
        ("Market state", obs.get("market_state")),
        ("Session extension", obs.get("session_extension")),
        ("Early opportunity", obs.get("early_opportunity")),
        ("Setups checked", obs.get("setups_checked")),
    ]
    lines = []
    for title, body in sections:
        text = str(body or "").strip()
        if text:
            lines.append(f"{title}:\n{text}")
    return "\n\n".join(lines)


def _build_observe_prompt(
    *,
    session: str,
    spot: float,
    bars_1m: int,
    bars_5m: int,
    bars_15m: int,
    bars_1h: int,
    entry_timeframe: str = "1m",
    entry_5m_fallback: bool = False,
) -> str:
    entry_label = "1-minute" if entry_timeframe == "1m" else "5-minute (1m unavailable)"
    entry_note = (
        "Image 1 is recent 5m bars (1m unavailable) — describe entry-timing cues only; "
        "primary trigger remains Image 2.\n"
        if entry_5m_fallback
        else "Image 1 is 1m entry timing — micro structure and confirmation vs 5m trigger.\n"
    )
    return (
        "You are an experienced XAUUSD chart analyst preparing a structured read for a "
        "scalper. OBSERVE ONLY — do NOT decide TAKE or SKIP and do NOT recommend a trade.\n\n"
        "You receive FOUR candlestick charts (green=bullish, red=bearish) plus numeric OHLC:\n"
        f"  Image 1: {entry_label} ({bars_1m} candles) — entry timing.\n"
        f"  Image 2: 5-minute ({bars_5m} candles) — PRIMARY scalp trigger timeframe.\n"
        f"  Image 3: 15-minute ({bars_15m} candles) — structure and zones.\n"
        f"  Image 4: 1-hour ({bars_1h} candles) — session bias/context ONLY.\n"
        f"{entry_note}"
        "Shaded zones on 5m/15m (if present) mark FVG, IFVG, and OB — cite them with prices.\n\n"
        f"Session: {session}. Spot reference: {spot:.2f}.\n\n"
        f"{_early_opportunity_guidance()}\n"
        "For each field, be specific with price levels visible on the charts:\n"
        "- entry_chart: last few candles, micro structure, entry-timing cues.\n"
        "- chart_5m: sweeps, FVG/OB/IFVG reactions, momentum, ORB, MSS — actionable NOW or not.\n"
        "- chart_15m: higher structure, zones, premium/discount vs range.\n"
        "- chart_1h: directional bias only (not an entry trigger).\n"
        "- key_levels: PDH/PDL, Asian range, EQH/EQL, session opens, recent swing H/L with prices.\n"
        "- market_state: trending vs chop, session character, what matters at spot now.\n"
        "- session_extension: is price pumped (near range high) or dumped (near range low)? "
        "Cite % in range and distance to high/low.\n"
        "- early_opportunity: BEST early scalp forming NOW — fade after pump, bounce after dump, "
        "or fresh momentum — is first 5m trigger live? cite prices.\n"
        "- setups_checked: liquidity sweep, ORB, FVG/OB/IFVG, momentum, liq grab+MSS, "
        "SESSION FADE (extended up/down) — each yes/no for live 5m trigger and one-line why.\n"
    )


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
    chart_observation: Optional[str] = None,
    review_lesson: Optional[str] = None,
    confluence_checklist: Optional[Dict[str, bool]] = None,
    confidence_threshold: int = 85,
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
    prompt = (
        "You are an experienced XAUUSD SCALPER — you hunt fast, session-local setups "
        "that can play out in minutes to a few hours. You are NOT a swing trader. "
        "You do not target multi-day moves, distant daily/weekly levels, or slow "
        "You are decisive and risk-first: skip unclear chop rather than force a trade, "
        "but ACT on clear early fades and first 5m reversal triggers when price is extended.\n\n"
    )
    if review_lesson:
        prompt += (
            "LATEST AI PERFORMANCE REVIEW (from your last full trade audit — follow these rules):\n"
            f"{review_lesson.strip()}\n\n"
        )
    if chart_observation:
        prompt += (
            "TWO-STEP DECISION:\n"
            "- You already completed a structured chart observation (below).\n"
            "- Verify it against the images, then decide TAKE or SKIP only.\n"
            "- In rationale, focus on the decision — do not repeat the full observation.\n\n"
            "PRIOR CHART OBSERVATION:\n"
            f"{chart_observation}\n\n"
        )
    prompt += (
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
        f"{_early_opportunity_guidance()}\n"
        "SCALP RISK RULES (mandatory for every TAKE):\n"
        f"- Stop loss: {sl_range} platform pips from entry (XAUUSD: 1 pip = $0.10 → "
        f"{min_sl:.0f} pips ≈ ${min_sl * 0.1:.1f}, {max_sl:.0f} pips ≈ ${max_sl * 0.1:.1f} max).\n"
        "- Take profit: risk:reward 1:1 to 2:1 only — no 3R+ swing targets.\n"
        "- Entry near current price — no \"limit at distant OB\" scalps; if price isn't "
        "there now with a 5m trigger, SKIP.\n"
        f"- If the setup cannot fit {sl_range} pip SL with 1–2R TP at current price, SKIP.\n\n"
        f"{confidence_scoring_prompt_block(confidence_threshold=confidence_threshold)}\n\n"
    )
    if confluence_checklist:
        prompt += (
            f"{format_confluence_block(confluence_checklist, confidence_threshold=confidence_threshold)}\n\n"
        )
    prompt += (
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
        "6. SESSION FADE / EARLY REVERSAL — price extended in today's range (pumped near "
        "highs or dumped near lows) and FIRST 5m trigger flips against the extension; "
        "enter early for snap-back toward mid-range (liquidity_grab_short/long, "
        "liq_sweep_bear/bull, sdp_bear/sdp_bull, eqh_sweep_bear, eql_sweep_bull).\n\n"
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
        "- On SESSION FADE / extended range: TAKE early when first 5m reversal is live — "
        "do not skip just because HTF still looks trending; fades are valid scalps.\n"
        f"- {setup_vocabulary_prompt_block()}\n"
        "- SKIP: no qualifying scalp, swing-style setup, or chop — say which scalp "
        "patterns you checked on 1m/5m/15m and why none qualify. Never use "
        'setup_type "unknown" — omit setup_type on SKIP.\n\n'
        f"If TAKE: setup_type (exact id), direction, entry near spot, stop_loss, "
        f"take_profit (1–2R, {sl_range} pip SL), "
        "confidence 0–100, rationale naming the scalp pattern and 5m/15m levels.\n\n"
        "If SKIP: action SKIP, null prices, confidence 0–100, rationale plain and specific."
    )
    return prompt


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


def _normalize_observation(raw: Dict[str, Any]) -> Dict[str, Any]:
    def _s(key: str) -> str:
        return str(raw.get(key) or "").strip()

    return {
        "entry_chart": _s("entry_chart"),
        "chart_5m": _s("chart_5m"),
        "chart_15m": _s("chart_15m"),
        "chart_1h": _s("chart_1h"),
        "key_levels": _s("key_levels"),
        "market_state": _s("market_state"),
        "session_extension": _s("session_extension"),
        "early_opportunity": _s("early_opportunity"),
        "setups_checked": _s("setups_checked"),
    }


def _normalize_decision(
    raw: Dict[str, Any],
    *,
    confluence_checklist: Optional[Dict[str, bool]] = None,
    confidence_threshold: int = 85,
) -> Dict[str, Any]:
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
    if action == "TAKE":
        if not direction:
            action = "SKIP"
        elif not is_approved_setup_type(setup_type):
            action = "SKIP"
            setup_type = None
            confidence = min(confidence, 40)
    elif action == "SKIP" and not setup_type:
        setup_type = None

    def _f(key: str) -> Optional[float]:
        v = raw.get(key)
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    out = {
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
    calibrated, meta = calibrate_confidence(
        out,
        checklist=confluence_checklist or {},
        confidence_threshold=confidence_threshold,
    )
    out["confidence"] = calibrated
    if meta.get("calibrated"):
        out["confidence_model"] = meta.get("model_confidence")
    out["confidence_meta"] = meta
    return out


def _build_ohlc_context(
    *,
    entry_bars: list,
    bars_5m: list,
    bars_15m: list,
    bars_1h: list,
    entry_timeframe: str,
    zone_summary: Optional[str] = None,
    spot: Optional[float] = None,
) -> str:
    entry_label = "1m" if entry_timeframe == "1m" else "5m-entry"
    parts = [
        "NUMERIC OHLC SUMMARY (cross-check against the chart images):",
        summarize_bars_for_prompt(entry_bars, label=f"Entry {entry_label}"),
        summarize_bars_for_prompt(bars_5m, label="5m"),
        summarize_bars_for_prompt(bars_15m, label="15m"),
        summarize_bars_for_prompt(bars_1h, label="1h"),
    ]
    if spot and spot > 0:
        ext = summarize_session_extension(bars_5m, bars_15m, spot)
        if ext:
            parts.append(ext)
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


async def _invoke_gemini_vision(
    *,
    client,
    cfg: GeminiGoldRuntimeConfig,
    genai_types,
    png_entry: bytes,
    png_5m: bytes,
    png_15m: bytes,
    png_1h: bytes,
    entry_bars: list,
    entry_timeframe: str,
    entry_5m_fallback: bool,
    bars_5m: list,
    bars_15m: list,
    bars_1h: list,
    prompt: str,
    response_schema: type[BaseModel],
    max_output_tokens: int,
    timeout_s: float,
    temperature: float = 0.2,
):
    def _call():
        return client.models.generate_content(
            model=cfg.model,
            contents=_chart_contents(
                genai_types,
                png_entry=png_entry,
                png_5m=png_5m,
                png_15m=png_15m,
                png_1h=png_1h,
                entry_timeframe=entry_timeframe,
                entry_5m_fallback=entry_5m_fallback,
                bars_1m=len(entry_bars),
                bars_5m=len(bars_5m),
                bars_15m=len(bars_15m),
                bars_1h=len(bars_1h),
                prompt=prompt,
            ),
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                response_mime_type="application/json",
                response_schema=response_schema,
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
            ),
        )

    try:
        response = await asyncio.wait_for(asyncio.to_thread(_call), timeout=timeout_s)
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

    return raw, tokens_in, tokens_out, cost, None


async def describe_charts(
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
    Step 1: structured chart observation (no TAKE/SKIP).
    Returns (observation_dict, tokens_in, tokens_out, cost_usd, error).
    """
    client = _get_gemini_client()
    if not client:
        return None, 0, 0, 0.0, "no_gemini_api_key"

    from google.genai import types as genai_types

    prompt = _build_observe_prompt(
        session=session,
        spot=spot,
        bars_1m=len(entry_bars),
        bars_5m=len(bars_5m),
        bars_15m=len(bars_15m),
        bars_1h=len(bars_1h),
        entry_timeframe=entry_timeframe,
        entry_5m_fallback=entry_5m_fallback,
    )
    ohlc_block = _build_ohlc_context(
        entry_bars=entry_bars,
        bars_5m=bars_5m,
        bars_15m=bars_15m,
        bars_1h=bars_1h,
        entry_timeframe=entry_timeframe,
        zone_summary=zone_summary,
        spot=spot,
    )
    full_prompt = f"{prompt}\n\n{ohlc_block}"

    raw, tokens_in, tokens_out, cost, err = await _invoke_gemini_vision(
        client=client,
        cfg=cfg,
        genai_types=genai_types,
        png_entry=png_1m,
        png_5m=png_5m,
        png_15m=png_15m,
        png_1h=png_1h,
        entry_bars=entry_bars,
        entry_timeframe=entry_timeframe,
        entry_5m_fallback=entry_5m_fallback,
        bars_5m=bars_5m,
        bars_15m=bars_15m,
        bars_1h=bars_1h,
        prompt=full_prompt,
        response_schema=GeminiGoldChartObservationSchema,
        max_output_tokens=900,
        timeout_s=_OBSERVE_TIMEOUT_S,
        temperature=0.15,
    )
    if err or not raw:
        return None, tokens_in, tokens_out, cost, err or "no_observation"
    return _normalize_observation(raw), tokens_in, tokens_out, cost, None


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
    chart_observation: Optional[str] = None,
    review_lesson: Optional[str] = None,
    confluence_checklist: Optional[Dict[str, bool]] = None,
) -> Tuple[Optional[Dict[str, Any]], int, int, float, Optional[str]]:
    """
    Step 2 (or single-step): TAKE/SKIP decision from chart images.
    Returns (decision_dict, tokens_in, tokens_out, cost_usd, error).
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
        chart_observation=chart_observation,
        review_lesson=review_lesson,
        confluence_checklist=confluence_checklist,
        confidence_threshold=cfg.confidence_threshold,
    )
    ohlc_block = _build_ohlc_context(
        entry_bars=entry_bars,
        bars_5m=bars_5m,
        bars_15m=bars_15m,
        bars_1h=bars_1h,
        entry_timeframe=entry_timeframe,
        zone_summary=zone_summary,
        spot=spot,
    )
    full_prompt = f"{prompt}\n\n{ohlc_block}"

    raw, tokens_in, tokens_out, cost, err = await _invoke_gemini_vision(
        client=client,
        cfg=cfg,
        genai_types=genai_types,
        png_entry=png_1m,
        png_5m=png_5m,
        png_15m=png_15m,
        png_1h=png_1h,
        entry_bars=entry_bars,
        entry_timeframe=entry_timeframe,
        entry_5m_fallback=entry_5m_fallback,
        bars_5m=bars_5m,
        bars_15m=bars_15m,
        bars_1h=bars_1h,
        prompt=full_prompt,
        response_schema=GeminiGoldDecisionSchema,
        max_output_tokens=512,
        timeout_s=_CALL_TIMEOUT_S,
        temperature=0.2,
    )
    if err or not raw:
        return None, tokens_in, tokens_out, cost, err or "no_decision"
    return _normalize_decision(
        raw,
        confluence_checklist=confluence_checklist,
        confidence_threshold=cfg.confidence_threshold,
    ), tokens_in, tokens_out, cost, None


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
    return _normalize_decision(
        raw,
        confluence_checklist=None,
        confidence_threshold=confidence_threshold,
    ), tokens_in, tokens_out, cost, None
