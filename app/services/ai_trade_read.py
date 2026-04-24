"""
Chart-aware AI trade read for /trade.

Builds a Claude prompt that incorporates EVERYTHING the user can see on the
chart at that moment — current price, recent price action, every visible
indicator with its current value, the cached order-book wall snapshot,
big-prints flow, and which overlays the user has toggled on — then forces a
strict structured trade plan with an explicit ORDER_TYPE (MARKET vs LIMIT).

This is intentionally separate from `liquidity_walls.scan_walls(use_ai=True)`,
which only sees order books. That summary still drives the wall panel; this
one drives the right-rail "AI trade read" card.
"""
from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── TA primitives (mirror of trade.html JS engine + alerts_engine.py) ────────
def _sma(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    return sum(values[-period:]) / period


def _ema(values: List[float], period: int) -> Optional[float]:
    if period <= 0 or len(values) < period:
        return None
    k = 2.0 / (period + 1)
    out = sum(values[:period]) / period
    for v in values[period:]:
        out = v * k + out * (1 - k)
    return out


def _rsi(values: List[float], period: int = 14) -> Optional[float]:
    if len(values) < period + 1:
        return None
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


def _macd(values: List[float], fast: int, slow: int, signal: int
          ) -> Optional[Tuple[float, float, float]]:
    """Returns (macd, signal, hist) for the latest bar, or None."""
    ef = _ema(values, fast)
    es = _ema(values, slow)
    if ef is None or es is None or len(values) < slow + signal:
        return None
    # Build MACD series across the tail and EMA-smooth it for the signal
    macd_series: List[float] = []
    # Recompute EMAs progressively to get the macd series
    k_fast = 2.0 / (fast + 1)
    k_slow = 2.0 / (slow + 1)
    f_seed = sum(values[:fast]) / fast
    s_seed = sum(values[:slow]) / slow
    ema_f = f_seed
    ema_s = s_seed
    for i, v in enumerate(values):
        if i >= fast:
            ema_f = v * k_fast + ema_f * (1 - k_fast)
        if i >= slow:
            ema_s = v * k_slow + ema_s * (1 - k_slow)
        if i >= slow - 1:
            macd_series.append(ema_f - ema_s)
    if len(macd_series) < signal:
        return None
    sig_seed = sum(macd_series[:signal]) / signal
    k_sig = 2.0 / (signal + 1)
    sig = sig_seed
    for v in macd_series[signal:]:
        sig = v * k_sig + sig * (1 - k_sig)
    macd_line = macd_series[-1]
    return (macd_line, sig, macd_line - sig)


def _bb(values: List[float], period: int, mult: float) -> Optional[Tuple[float, float, float]]:
    """Returns (lower, mid, upper)."""
    mid = _sma(values, period)
    if mid is None:
        return None
    window = values[-period:]
    var = sum((v - mid) ** 2 for v in window) / period
    sd = math.sqrt(var)
    return (mid - mult * sd, mid, mid + mult * sd)


def _atr(candles: List[dict], period: int) -> Optional[float]:
    if len(candles) < period + 1:
        return None
    trs: List[float] = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]; l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    atr = sum(trs[:period]) / period
    for v in trs[period:]:
        atr = (atr * (period - 1) + v) / period
    return atr


def _vwap(candles: List[dict]) -> Optional[float]:
    """Session VWAP — uses today's UTC bars only."""
    if not candles:
        return None
    last_ts = candles[-1].get("time", 0)
    if not last_ts:
        return None
    day_start = last_ts - (last_ts % 86400)
    pv = 0.0
    vv = 0.0
    for c in candles:
        if (c.get("time") or 0) < day_start:
            continue
        typical = (c["high"] + c["low"] + c["close"]) / 3.0
        vol = c.get("volume", 0) or 0
        pv += typical * vol
        vv += vol
    if vv <= 0:
        return None
    return pv / vv


def _supertrend(candles: List[dict], period: int, mult: float
                ) -> Optional[Tuple[int, float]]:
    """Returns (direction +1/-1, current trend line value)."""
    n = len(candles)
    if n < period + 2:
        return None
    tr: List[float] = []
    for i, c in enumerate(candles):
        if i == 0:
            tr.append(c["high"] - c["low"])
        else:
            p = candles[i - 1]
            tr.append(max(
                c["high"] - c["low"],
                abs(c["high"] - p["close"]),
                abs(c["low"] - p["close"]),
            ))
    atr_arr: List[Optional[float]] = [None] * n
    seed = sum(tr[:period]) / period
    atr_arr[period - 1] = seed
    for i in range(period, n):
        atr_arr[i] = (atr_arr[i - 1] * (period - 1) + tr[i]) / period  # type: ignore

    upper: List[Optional[float]] = [None] * n
    lower: List[Optional[float]] = [None] * n
    direction = 1
    for i in range(n):
        c = candles[i]
        if atr_arr[i] is None:
            continue
        hl2 = (c["high"] + c["low"]) / 2.0
        b_up = hl2 + mult * atr_arr[i]  # type: ignore
        b_dn = hl2 - mult * atr_arr[i]  # type: ignore
        if i > 0 and upper[i - 1] is not None:
            if b_up > upper[i - 1] and candles[i - 1]["close"] <= upper[i - 1]:
                b_up = min(b_up, upper[i - 1])  # type: ignore
            if b_dn < lower[i - 1] and candles[i - 1]["close"] >= lower[i - 1]:
                b_dn = max(b_dn, lower[i - 1])  # type: ignore
        upper[i] = b_up
        lower[i] = b_dn
        if direction <= 0 and c["close"] > (upper[i - 1] if i > 0 and upper[i - 1] else b_up):
            direction = 1
        elif direction >= 0 and c["close"] < (lower[i - 1] if i > 0 and lower[i - 1] else b_dn):
            direction = -1
    last_line = lower[-1] if direction > 0 else upper[-1]
    return (direction, float(last_line) if last_line is not None else 0.0)


def _stoch_rsi(values: List[float], period: int, stoch: int
               ) -> Optional[float]:
    """Returns the %K value of StochRSI (0–100) for the latest bar."""
    # First build RSI series across the tail
    if len(values) < period + stoch + 1:
        return None
    rsi_series: List[float] = []
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_g = sum(gains[:period]) / period
    avg_l = sum(losses[:period]) / period
    rsi_series.append(100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l))
    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i]) / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rsi_series.append(100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l))
    if len(rsi_series) < stoch:
        return None
    window = rsi_series[-stoch:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 50.0
    return 100.0 * (rsi_series[-1] - lo) / (hi - lo)


# ─── Indicator describer ──────────────────────────────────────────────────────
def _src_values(candles: List[dict], src: str) -> List[float]:
    src = (src or "close").lower()
    if src == "open":   return [c["open"] for c in candles]
    if src == "high":   return [c["high"] for c in candles]
    if src == "low":    return [c["low"]  for c in candles]
    if src == "hl2":    return [(c["high"] + c["low"]) / 2.0 for c in candles]
    if src == "hlc3":   return [(c["high"] + c["low"] + c["close"]) / 3.0 for c in candles]
    if src == "ohlc4":  return [(c["open"] + c["high"] + c["low"] + c["close"]) / 4.0 for c in candles]
    return [c["close"] for c in candles]


def _fmt_price(v: float) -> str:
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:,.4f}"
    return f"{v:.6f}"


def describe_indicator(spec: Dict[str, Any], candles: List[dict],
                       last_price: float) -> Optional[str]:
    """Return a single human/Claude line for one indicator, or None if skipped."""
    t = (spec.get("type") or "").lower()
    p = spec.get("params") or {}
    src = (spec.get("src") or "close").lower()
    closes = [c["close"] for c in candles]
    sv = _src_values(candles, src)

    try:
        if t == "ema":
            n = int(p.get("period", 20))
            v = _ema(sv, n)
            if v is None: return None
            d = (last_price - v) / v * 100
            tag = "above" if d > 0 else "below"
            return f"EMA({n}, {src}) = {_fmt_price(v)}  →  price {abs(d):.2f}% {tag}"
        if t == "sma":
            n = int(p.get("period", 20))
            v = _sma(sv, n)
            if v is None: return None
            d = (last_price - v) / v * 100
            tag = "above" if d > 0 else "below"
            return f"SMA({n}, {src}) = {_fmt_price(v)}  →  price {abs(d):.2f}% {tag}"
        if t == "rsi":
            n = int(p.get("period", 14))
            v = _rsi(sv, n)
            if v is None: return None
            zone = ("overbought" if v >= 70 else "oversold" if v <= 30
                    else "bullish bias" if v > 55 else "bearish bias" if v < 45
                    else "neutral")
            return f"RSI({n}) = {v:.1f}  ({zone})"
        if t == "macd":
            f = int(p.get("fast", 12)); sl = int(p.get("slow", 26)); sg = int(p.get("signal", 9))
            res = _macd(sv, f, sl, sg)
            if res is None: return None
            macd, sig, hist = res
            zero = "above zero (bullish)" if macd > 0 else "below zero (bearish)"
            cross = "bull stack (macd>sig)" if macd > sig else "bear stack (macd<sig)"
            return f"MACD({f},{sl},{sg}) = {macd:+.2f}, sig {sig:+.2f}, hist {hist:+.2f} ({zero}, {cross})"
        if t == "bb":
            n = int(p.get("period", 20)); m = float(p.get("mult", 2))
            res = _bb(sv, n, m)
            if res is None: return None
            lo, mid, hi = res
            band_w = (hi - lo) / mid * 100 if mid else 0
            pos = (last_price - lo) / (hi - lo) if hi > lo else 0.5
            zone = ("riding upper (extended)" if pos > 0.85
                    else "riding lower (extended down)" if pos < 0.15
                    else "above middle" if pos > 0.5 else "below middle")
            return (f"BB({n},{m}) = lo {_fmt_price(lo)} / mid {_fmt_price(mid)} / hi {_fmt_price(hi)} "
                    f"width {band_w:.2f}%, price at {pos*100:.0f}% of band ({zone})")
        if t == "vwap":
            v = _vwap(candles)
            if v is None: return None
            d = (last_price - v) / v * 100
            tag = "above" if d > 0 else "below"
            return f"VWAP (session) = {_fmt_price(v)}  →  price {abs(d):.2f}% {tag}"
        if t == "atr":
            n = int(p.get("period", 14))
            v = _atr(candles, n)
            if v is None: return None
            d = v / last_price * 100 if last_price else 0
            return f"ATR({n}) = {_fmt_price(v)}  ({d:.2f}% of price)"
        if t == "supertrend":
            n = int(p.get("period", 10)); m = float(p.get("mult", 3))
            res = _supertrend(candles, n, m)
            if res is None: return None
            direction, line = res
            arrow = "LONG" if direction > 0 else "SHORT"
            return f"SuperTrend({n},{m}) = {arrow} @ {_fmt_price(line)}"
        if t == "stochrsi":
            n = int(p.get("period", 14)); st = int(p.get("stoch", 14))
            v = _stoch_rsi(sv, n, st)
            if v is None: return None
            zone = "overbought" if v >= 80 else "oversold" if v <= 20 else "neutral"
            return f"StochRSI({n},{st}) %K = {v:.1f}  ({zone})"
    except Exception as e:
        logger.debug(f"indicator describe failed for {t}: {e}")
        return None
    return None


# ─── Price-action snapshot ────────────────────────────────────────────────────
def price_action_block(candles: List[dict], tf: str) -> Dict[str, Any]:
    if not candles:
        return {}
    closes = [c["close"] for c in candles]
    highs = [c["high"] for c in candles]
    lows = [c["low"] for c in candles]
    last = closes[-1]
    # Bars per hour for this TF
    bars_per_hour = {"1m": 60, "5m": 12, "15m": 4, "1h": 1}.get(tf, 12)
    one_hr = closes[-bars_per_hour - 1] if len(closes) > bars_per_hour else closes[0]
    four_hr = closes[-bars_per_hour * 4 - 1] if len(closes) > bars_per_hour * 4 else closes[0]
    one_h_pct = (last - one_hr) / one_hr * 100 if one_hr else 0
    four_h_pct = (last - four_hr) / four_hr * 100 if four_hr else 0
    # Recent swing high/low across last ~60 bars (or all available)
    look = min(len(candles), 60)
    sw_hi = max(highs[-look:])
    sw_lo = min(lows[-look:])
    # Last 8 candle directions as a string (G = green, R = red, . = doji-ish)
    dir_str = ""
    for c in candles[-8:]:
        body = c["close"] - c["open"]
        rng = c["high"] - c["low"]
        if rng > 0 and abs(body) / rng < 0.1:
            dir_str += "."
        elif body > 0:
            dir_str += "G"
        else:
            dir_str += "R"
    return {
        "last": last,
        "tf": tf,
        "pct_1h": one_h_pct,
        "pct_4h": four_h_pct,
        "swing_hi": sw_hi,
        "swing_lo": sw_lo,
        "swing_hi_dist_pct": (sw_hi - last) / last * 100 if last else 0,
        "swing_lo_dist_pct": (last - sw_lo) / last * 100 if last else 0,
        "recent_dir": dir_str,
    }


# ─── Wall context block (trimmed from a wall report) ──────────────────────────
def wall_context_block(report: Optional[dict]) -> str:
    if not report:
        return "No wall data available."
    price = report.get("price", 0)
    pl = report.get("pressure_label", "n/a")
    ps = report.get("pressure_score", 0.0)

    def _fmt_w(w: dict) -> str:
        side = w.get("side", "?")
        return (f"{side.upper()} @ {_fmt_price(w.get('price', 0))} "
                f"(${w.get('size_usd', 0):,.0f}, "
                f"{w.get('distance_pct', 0):+.2f}% away, "
                f"conf {w.get('confidence', 0):.2f})")

    top_buys = (report.get("top_buys") or [])[:3]
    top_sells = (report.get("top_sells") or [])[:3]
    parts = [
        f"Spot mid: {_fmt_price(price)}",
        f"Pressure: {pl} ({ps:+.2f})",
    ]
    if top_buys:
        parts.append("Top buy walls (support):")
        parts.extend("  - " + _fmt_w(w) for w in top_buys)
    if top_sells:
        parts.append("Top sell walls (resistance):")
        parts.extend("  - " + _fmt_w(w) for w in top_sells)
    return "\n".join(parts)


# ─── Tape (big prints) summary ────────────────────────────────────────────────
def tape_summary_block(tape: Dict[str, Any]) -> str:
    if not tape:
        return "Big-prints panel is off or has no recent prints."
    bn = int(tape.get("buy_count", 0))
    sn = int(tape.get("sell_count", 0))
    bu = float(tape.get("buy_usd", 0))
    su = float(tape.get("sell_usd", 0))
    if bn + sn == 0:
        return "Big-prints panel on, but no qualifying prints in the recent window."
    delta = bu - su
    bias = ("aggressive buying" if delta > su * 0.25
            else "aggressive selling" if -delta > bu * 0.25
            else "balanced two-way flow")
    return (f"Recent big-prints flow ({bn + sn} prints): "
            f"buys ${bu:,.0f} ({bn}) vs sells ${su:,.0f} ({sn})  →  "
            f"net ${delta:+,.0f} — {bias}.")


# ─── Toggle context ───────────────────────────────────────────────────────────
def toggles_block(toggles: Dict[str, Any]) -> str:
    bits = []
    if toggles.get("order_blocks"):
        bits.append("Order Blocks ON (user is watching SMC zones overlaid on chart)")
    if toggles.get("liq_heatmap"):
        bits.append("Liq Heatmap ON (user is watching where stops cluster)")
    if toggles.get("big_prints"):
        bits.append("Big Prints ON (user is watching aggressive market orders ≥ threshold)")
    if not bits:
        return "User has all chart overlays disabled."
    return "User has these overlays ON: " + "; ".join(bits) + "."


# ─── Main entry point ─────────────────────────────────────────────────────────
async def generate_ai_trade_read(
    *,
    symbol: str,
    tf: str,
    candles: List[dict],
    indicators: List[Dict[str, Any]],
    toggles: Dict[str, Any],
    tape: Dict[str, Any],
    wall_report: Optional[dict],
) -> Dict[str, Any]:
    """Compose the prompt, call Claude, return {summary, fallback, sources_used}.

    Always returns a structured plan — uses a deterministic fallback if Claude
    is unreachable so the panel never goes blank.
    """
    if not candles or len(candles) < 30:
        return {
            "summary": "Need more candle history to read this chart. Try again in a moment.",
            "fallback": True,
            "sources_used": [],
        }

    last_price = candles[-1]["close"]
    pa = price_action_block(candles, tf)

    indicator_lines: List[str] = []
    for spec in indicators or []:
        line = describe_indicator(spec, candles, last_price)
        if line:
            indicator_lines.append("  - " + line)
    if not indicator_lines:
        indicator_lines = ["  - (no indicators on the chart)"]

    sources = ["candles", f"{len(indicator_lines)} indicators"]
    if wall_report: sources.append("walls")
    if tape:        sources.append("tape")
    if toggles:     sources.append("toggles")

    prompt = (
        f"You are a degen crypto futures scalper running 50x-200x leverage on {symbol}.\n"
        f"You are reading the user's LIVE chart and must give them ONE actionable trade — "
        f"never 'wait', never 'no setup', never 'flat'. If conditions aren't great for a "
        f"market entry, give a LIMIT order at the best level (a wall, an indicator, a swing "
        f"point). Always pick the higher-probability side based on the full picture below.\n\n"

        f"=== PRICE ACTION ({tf} chart) ===\n"
        f"Last price: {_fmt_price(pa['last'])}  ({pa['pct_1h']:+.2f}% 1h, {pa['pct_4h']:+.2f}% 4h)\n"
        f"Recent swing high: {_fmt_price(pa['swing_hi'])} ({pa['swing_hi_dist_pct']:+.2f}% from price)\n"
        f"Recent swing low:  {_fmt_price(pa['swing_lo'])} ({-pa['swing_lo_dist_pct']:+.2f}% from price)\n"
        f"Last 8 candles: {pa['recent_dir']}  (G=green R=red .=doji)\n\n"

        f"=== INDICATORS ON CHART ({len(indicator_lines)} active) ===\n"
        + "\n".join(indicator_lines) + "\n\n"

        f"=== ORDER BOOK / WALLS ===\n"
        f"{wall_context_block(wall_report)}\n\n"

        f"=== AGGRESSIVE FLOW ===\n"
        f"{tape_summary_block(tape)}\n\n"

        f"=== USER'S CHART OVERLAYS ===\n"
        f"{toggles_block(toggles)}\n\n"

        f"=== OUTPUT RULES ===\n"
        f"Output MUST follow this EXACT structure with these EXACT labels — no prose paragraphs, "
        f"no greetings, no emojis, no markdown. ALL fields required.\n\n"
        f"BIAS: <one short sentence — which side dominates and the magnet level. "
        f"Weight indicator stack + wall pressure + flow.>\n"
        f"TRADE: <LONG or SHORT> @ <price>\n"
        f"ORDER_TYPE: <MARKET — short reason> or <LIMIT — short reason (e.g. 'rest at the FRESH BIG buy wall')>\n"
        f"STOP: <price> (<distance %>)\n"
        f"TP1: <price> (<distance %>)\n"
        f"TP2: <price> (<distance %>)\n"
        f"R:R: <ratio like 1:2.5  using TP1>\n"
        f"LEVERAGE: <safe lev like '100x ok' or '50x max — stop too tight for 100x' or '200x ok'>\n"
        f"INVALIDATION: <one price level. If price breaks this, setup is dead — cut.>\n"
        f"ODDS: UP <int>% / DOWN <int>% / CHOP <int>%   (must sum to 100, be opinionated)\n"
        f"NOTE: <one short line — the single biggest signal you weighted, or the biggest risk>\n"
    )

    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return {
            "summary": _fallback_plan(symbol, pa, wall_report),
            "fallback": True,
            "sources_used": sources,
        }

    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=420,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (msg.content[0].text or "").strip() if msg.content else ""
        if not text:
            raise RuntimeError("empty response")
        return {"summary": text, "fallback": False, "sources_used": sources}
    except Exception as e:
        logger.warning(f"AI trade read failed for {symbol}: {e}")
        return {
            "summary": _fallback_plan(symbol, pa, wall_report),
            "fallback": True,
            "sources_used": sources,
        }


def _fallback_plan(symbol: str, pa: Dict[str, Any], wall_report: Optional[dict]) -> str:
    """Deterministic plan if Claude is unreachable — keeps the panel useful."""
    last = pa.get("last", 0)
    sw_hi = pa.get("swing_hi", last)
    sw_lo = pa.get("swing_lo", last)
    pct_1h = pa.get("pct_1h", 0)
    side = "LONG" if pct_1h >= 0 else "SHORT"
    if side == "LONG":
        entry = sw_lo + (last - sw_lo) * 0.5
        stop = sw_lo * 0.998
        tp1 = sw_hi
    else:
        entry = sw_hi - (sw_hi - last) * 0.5
        stop = sw_hi * 1.002
        tp1 = sw_lo
    risk = abs(entry - stop)
    reward = abs(tp1 - entry)
    rr = reward / risk if risk else 0
    return (
        f"BIAS: 1h trend is {pct_1h:+.2f}%, biased {side.lower()}.\n"
        f"TRADE: {side} @ {_fmt_price(entry)}\n"
        f"ORDER_TYPE: LIMIT — AI offline, resting at midpoint between current price and last swing\n"
        f"STOP: {_fmt_price(stop)} ({abs(stop-entry)/entry*100:.2f}%)\n"
        f"TP1: {_fmt_price(tp1)} ({abs(tp1-entry)/entry*100:.2f}%)\n"
        f"TP2: {_fmt_price(tp1)} ({abs(tp1-entry)/entry*100:.2f}%)\n"
        f"R:R: 1:{rr:.2f}\n"
        f"LEVERAGE: 25x — AI fallback, conservative\n"
        f"INVALIDATION: {_fmt_price(stop)}\n"
        f"ODDS: UP 40% / DOWN 40% / CHOP 20%\n"
        f"NOTE: AI offline — this is a deterministic fallback based on swing structure.\n"
    )
