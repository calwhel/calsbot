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
    if toggles.get("fvg"):
        bits.append("FVG Zones ON (user is watching unfilled fair-value gaps as magnets)")
    if not bits:
        return "User has all chart overlays disabled."
    return "User has these overlays ON: " + "; ".join(bits) + "."


# ─── FVG context (ICT fair-value gaps + retest detection) ─────────────────────
def fvg_context_block(candles: List[dict], last_price: float) -> str:
    """List the nearest unfilled bull FVGs below price (support pockets) and
    bear FVGs above price (resistance pockets), plus a retest flag if the
    current bar is wicking into one. Mirrors the on-chart FVG overlay so the
    AI sees what the user sees."""
    try:
        from app.services.auto_trader import detect_fvgs, _fvg_retest_signal
    except Exception as e:
        logger.debug(f"fvg_context_block import failed: {e}")
        return "FVG data unavailable."
    if not candles or len(candles) < 30:
        return "Not enough history to detect FVGs."
    try:
        gaps = detect_fvgs(
            candles,
            min_gap_atr_mult=0.10,
            disp_atr_mult=0.5,
            only_unfilled=True,
            max_age_bars=200,
            max_results=30,
        )
    except Exception as e:
        logger.debug(f"detect_fvgs failed: {e}")
        return "FVG detector errored."
    if not gaps:
        return "No active (unfilled) FVG zones in the last ~200 bars."

    # Bull FVGs that sit BELOW current price act as support / long magnets.
    bulls_below = sorted(
        [g for g in gaps if g["side"] == "bull" and g["top"] <= last_price],
        key=lambda g: last_price - g["top"],
    )[:3]
    # Bear FVGs that sit ABOVE current price act as resistance / short magnets.
    bears_above = sorted(
        [g for g in gaps if g["side"] == "bear" and g["bottom"] >= last_price],
        key=lambda g: g["bottom"] - last_price,
    )[:3]

    parts: List[str] = []
    # Active retest is the strongest signal — surface it first.
    try:
        side, note = _fvg_retest_signal(candles)
    except Exception:
        side, note = None, ""
    if side:
        parts.append(f"⚡ ACTIVE FVG RETEST → {side.upper()} bias — {note}")

    if bulls_below:
        parts.append("Unfilled BULL FVGs below price (support / long magnets):")
        for g in bulls_below:
            dist = (last_price - g["top"]) / last_price * 100 if last_price else 0
            parts.append(
                f"  - {_fmt_price(g['bottom'])}–{_fmt_price(g['top'])}  "
                f"(mid {_fmt_price(g['mid'])}, {g['size_pct']:.2f}% wide, "
                f"{g['size_atr']:.2f}×ATR, {dist:.2f}% below price, "
                f"{g['age_bars']} bars old)"
            )
    if bears_above:
        parts.append("Unfilled BEAR FVGs above price (resistance / short magnets):")
        for g in bears_above:
            dist = (g["bottom"] - last_price) / last_price * 100 if last_price else 0
            parts.append(
                f"  - {_fmt_price(g['bottom'])}–{_fmt_price(g['top'])}  "
                f"(mid {_fmt_price(g['mid'])}, {g['size_pct']:.2f}% wide, "
                f"{g['size_atr']:.2f}×ATR, {dist:.2f}% above price, "
                f"{g['age_bars']} bars old)"
            )
    if not bulls_below and not bears_above and not side:
        return "No FVGs near price (closest gaps are stale or already filled)."
    return "\n".join(parts)


# ─── Higher-timeframe trend block ─────────────────────────────────────────────
def _htf_summary(candles: List[dict], label: str, last_price: float) -> Optional[str]:
    """Compact HTF read for one timeframe — SMA(20)/SMA(50) stack, slope, and
    last 5 closed bars' direction string."""
    if not candles or len(candles) < 55:
        return None
    closes = [c["close"] for c in candles]
    sma20 = _sma(closes, 20)
    sma50 = _sma(closes, 50)
    if sma20 is None or sma50 is None:
        return None
    # Slope of SMA(50) — compare current vs 5 bars ago. Avoids noise.
    sma50_5ago = _sma(closes[:-5], 50) if len(closes) > 55 else sma50
    slope_pct = ((sma50 - (sma50_5ago or sma50)) / (sma50_5ago or sma50) * 100
                 if sma50_5ago else 0.0)
    if sma20 > sma50 and slope_pct > 0.05:
        bias = "BULL stack (SMA20>SMA50, slope rising)"
    elif sma20 < sma50 and slope_pct < -0.05:
        bias = "BEAR stack (SMA20<SMA50, slope falling)"
    else:
        bias = "RANGE/transition (no clean stack)"
    px_vs_50 = (last_price - sma50) / sma50 * 100 if sma50 else 0
    side_word = "above" if px_vs_50 > 0 else "below"
    # Last 5 HTF candle directions
    dir_str = ""
    for c in candles[-5:]:
        body = c["close"] - c["open"]
        rng = c["high"] - c["low"]
        if rng > 0 and abs(body) / rng < 0.1: dir_str += "."
        elif body > 0: dir_str += "G"
        else: dir_str += "R"
    return (f"{label}: {bias}. SMA20={_fmt_price(sma20)}, SMA50={_fmt_price(sma50)} "
            f"(slope {slope_pct:+.2f}% over 5 bars). Price {abs(px_vs_50):.2f}% "
            f"{side_word} SMA50. Last 5 bars: {dir_str}")


def htf_trend_block(htf_1h: List[dict], htf_4h: List[dict],
                    last_price: float) -> str:
    lines: List[str] = []
    s1 = _htf_summary(htf_1h, "1H", last_price) if htf_1h else None
    s4 = _htf_summary(htf_4h, "4H", last_price) if htf_4h else None
    if s1: lines.append(s1)
    if s4: lines.append(s4)
    if not lines:
        return "HTF data unavailable."
    # Confluence read across timeframes
    bull_count = sum(1 for s in (s1, s4) if s and "BULL stack" in s)
    bear_count = sum(1 for s in (s1, s4) if s and "BEAR stack" in s)
    if bull_count == 2:
        lines.append("→ HTF CONFLUENCE: both 1H and 4H bullish. Counter-trend shorts are scalps only.")
    elif bear_count == 2:
        lines.append("→ HTF CONFLUENCE: both 1H and 4H bearish. Counter-trend longs are scalps only.")
    elif bull_count == 1 and bear_count == 1:
        lines.append("→ HTF CONFLICT: 1H and 4H disagree — favour mean-reversion to the conflicted level.")
    return "\n".join(lines)


# ─── Funding rate / open-interest context ─────────────────────────────────────
def funding_oi_block(funding: Optional[dict]) -> str:
    """Render Coinglass funding + OI snapshot. The funding_rate_pct value is a
    raw % per 8h (Binance convention), e.g. 0.012 = +0.012% per 8h."""
    if not funding:
        return "Funding/OI data unavailable."
    fr = funding.get("funding_rate_pct")
    oi = funding.get("open_interest_usd")
    oi_chg = funding.get("oi_change_24h_pct")
    parts: List[str] = []
    if fr is not None:
        try:
            frv = float(fr)
        except (TypeError, ValueError):
            frv = 0.0
        if frv >= 0.05:
            tag = "EXTREME LONG (over-leveraged longs — squeeze risk)"
        elif frv >= 0.02:
            tag = "elevated long bias"
        elif frv <= -0.03:
            tag = "EXTREME SHORT (over-leveraged shorts — squeeze risk)"
        elif frv <= -0.01:
            tag = "elevated short bias"
        else:
            tag = "neutral funding"
        ex = funding.get("funding_exchange") or "agg"
        parts.append(f"Funding: {frv:+.4f}% per 8h ({ex}) — {tag}")
    if oi is not None:
        try:
            oiv = float(oi)
            parts.append(f"Open Interest: ${oiv:,.0f}")
        except (TypeError, ValueError):
            pass
    if oi_chg is not None:
        try:
            oc = float(oi_chg)
            tag = ("rising — fresh positions opening (trend-confirming)" if oc > 5
                   else "falling — positions closing (de-risking / trend-fading)" if oc < -5
                   else "flat")
            parts.append(f"OI 24h Δ: {oc:+.2f}% — {tag}")
        except (TypeError, ValueError):
            pass
    if not parts:
        return "Funding/OI data unavailable."
    return "\n".join(parts)


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
    htf_1h_candles: Optional[List[dict]] = None,
    htf_4h_candles: Optional[List[dict]] = None,
    funding_data: Optional[dict] = None,
) -> Dict[str, Any]:
    """Compose the prompt, call Claude, return {summary, fallback, sources_used}.

    Always returns a structured plan — uses a deterministic fallback if Claude
    is unreachable so the panel never goes blank.

    HTF candles + funding are optional but strongly recommended; without them
    the prompt loses the macro confluence layer that prevents counter-trend
    chasing and funding-fade traps.
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

    fvg_text     = fvg_context_block(candles, last_price)
    htf_text     = htf_trend_block(htf_1h_candles or [], htf_4h_candles or [], last_price)
    funding_text = funding_oi_block(funding_data)

    sources = ["candles", f"{len(indicator_lines)} indicators"]
    if wall_report:                        sources.append("walls")
    if tape:                               sources.append("tape")
    if toggles:                            sources.append("toggles")
    if "No active" not in fvg_text and "unavailable" not in fvg_text:
                                           sources.append("fvg")
    if "unavailable" not in htf_text:      sources.append("htf")
    if "unavailable" not in funding_text:  sources.append("funding")

    prompt = (
        f"You are an elite crypto futures trader on {symbol}. You read the user's "
        f"LIVE chart plus higher-timeframe context, order-book walls, FVG zones, "
        f"funding/OI, and aggressive tape. You give ONE actionable trade — "
        f"never 'wait', never 'no setup', never 'flat'. If conditions aren't great "
        f"for market entry, give a LIMIT order at the best level (an unfilled FVG, "
        f"a fresh wall, an indicator, a swing). Pick the higher-probability side "
        f"based on the FULL picture — and you MUST cite which signals drove it.\n\n"

        f"=== HOW TO WEIGHT SIGNALS (in priority order) ===\n"
        f"1. ACTIVE FVG RETEST — if price is currently wicking into an unfilled FVG, "
        f"that side wins unless HTF strongly disagrees. This is the highest-EV setup.\n"
        f"2. HTF CONFLUENCE — if 1H + 4H both agree, only take WITH them on this TF "
        f"(counter-trend trades are scalps with tight stops, not full positions).\n"
        f"3. WALL PROXIMITY — if a fresh BIG wall is within 0.3%, prefer LIMIT entry "
        f"AT the wall with stop just past it; never market through a wall against you.\n"
        f"4. FUNDING EXTREMES — if funding is EXTREME LONG, longs need confirmation "
        f"(no chasing); shorts get a tailwind from squeeze risk. Mirror for short.\n"
        f"5. OI 24h — rising OI confirms the trending move; falling OI = de-risking, "
        f"prefer mean-reversion / fade.\n"
        f"6. INDICATORS + TAPE — tiebreakers, not lead signals.\n"
        f"Stops should anchor to STRUCTURE (opposite side of FVG, wall, swing) — never "
        f"a flat % unless nothing structural exists. TPs should target the next FVG, "
        f"wall, or HTF level — not arbitrary multiples.\n\n"

        f"=== HIGHER-TIMEFRAME TREND ===\n"
        f"{htf_text}\n\n"

        f"=== PRICE ACTION ({tf} chart — the user's view) ===\n"
        f"Last price: {_fmt_price(pa['last'])}  ({pa['pct_1h']:+.2f}% 1h, {pa['pct_4h']:+.2f}% 4h)\n"
        f"Recent swing high: {_fmt_price(pa['swing_hi'])} ({pa['swing_hi_dist_pct']:+.2f}% from price)\n"
        f"Recent swing low:  {_fmt_price(pa['swing_lo'])} ({-pa['swing_lo_dist_pct']:+.2f}% from price)\n"
        f"Last 8 candles: {pa['recent_dir']}  (G=green R=red .=doji)\n\n"

        f"=== FAIR VALUE GAPS (ICT — unfilled, ATR-quality-weighted) ===\n"
        f"{fvg_text}\n\n"

        f"=== ORDER BOOK / WALLS ===\n"
        f"{wall_context_block(wall_report)}\n\n"

        f"=== FUNDING / OPEN INTEREST ===\n"
        f"{funding_text}\n\n"

        f"=== INDICATORS ON CHART ({len(indicator_lines)} active) ===\n"
        + "\n".join(indicator_lines) + "\n\n"

        f"=== AGGRESSIVE FLOW ===\n"
        f"{tape_summary_block(tape)}\n\n"

        f"=== USER'S CHART OVERLAYS ===\n"
        f"{toggles_block(toggles)}\n\n"

        f"=== OUTPUT RULES ===\n"
        f"Output MUST follow this EXACT structure with these EXACT labels — no prose paragraphs, "
        f"no greetings, no emojis, no markdown. ALL fields required.\n\n"
        f"BIAS: <one sentence — which side wins AND name the 2-3 specific signals "
        f"driving it (e.g. '4H bull stack + active bull FVG retest at 67800 + sell-side "
        f"wall thin above'). NO generic phrases.>\n"
        f"TRADE: <LONG or SHORT> @ <price>\n"
        f"ORDER_TYPE: <MARKET — short reason> or <LIMIT — short reason (e.g. "
        f"'rest at unfilled bull FVG mid 67500' or 'rest at $1.2M buy wall 67200')>\n"
        f"STOP: <price> (<distance %>) — must anchor to STRUCTURE; name what "
        f"(e.g. 'below FVG bottom' / 'past the buy wall' / 'under swing low').\n"
        f"TP1: <price> (<distance %>) — name the target (next FVG / wall / swing / HTF level).\n"
        f"TP2: <price> (<distance %>) — name the target.\n"
        f"R:R: <ratio like 1:2.5  using TP1>\n"
        f"LEVERAGE: <safe lev like '100x ok' or '50x max — stop too wide' or '200x ok'>\n"
        f"INVALIDATION: <one price level. If price breaks this, setup is dead — cut.>\n"
        f"ODDS: UP <int>% / DOWN <int>% / CHOP <int>%   (must sum to 100, be opinionated)\n"
        f"NOTE: <one line — the single biggest signal you weighted AND the single "
        f"biggest risk. Both required.>\n"
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
        # Surface a structured error so callers (auto_trader) can react. We
        # specifically flag the "credit balance too low" case so the engine
        # can auto-pause strategies instead of hammering the API at $0.01/call.
        err_str = str(e).lower()
        ai_error = "unknown"
        if "credit balance" in err_str or "insufficient" in err_str:
            ai_error = "insufficient_credits"
        elif "rate" in err_str and "limit" in err_str:
            ai_error = "rate_limit"
        elif "401" in err_str or "auth" in err_str:
            ai_error = "auth"
        logger.warning(f"AI trade read failed for {symbol} ({ai_error}): {e}")
        return {
            "summary": _fallback_plan(symbol, pa, wall_report),
            "fallback": True,
            "ai_error": ai_error,
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
