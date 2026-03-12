"""
Strategy Condition Evaluators — Build Your Own Strategy Portal

Evaluates user-defined strategy conditions against live market data.
Supports: indicator, price_momentum, volume_spike, support_resistance, fvg, candlestick
All functions are pure (no side effects) — they take data, return True/False + detail.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cmp(actual: float, operator: str, threshold: float) -> bool:
    ops = {
        "gt": actual > threshold,
        "gte": actual >= threshold,
        "lt": actual < threshold,
        "lte": actual <= threshold,
        "eq": abs(actual - threshold) < 0.001,
    }
    return ops.get(operator, False)


# ─────────────────────────────────────────────────────────────────────────────
# Indicator conditions  (RSI, MACD, EMA, BB, Volume, SuperTrend, VWAP)
# ─────────────────────────────────────────────────────────────────────────────

def eval_indicator(cond: Dict, price_data: Dict, enhanced_ta: Dict) -> Tuple[bool, str]:
    """
    Evaluate a standard indicator condition.
    cond keys: name, timeframe, operator, value, [crossover_direction]
    """
    name = cond.get("name", "").lower()
    op   = cond.get("operator", "gt")
    val  = float(cond.get("value", 0))

    # RSI
    if name == "rsi":
        tf  = cond.get("timeframe", "15m")
        key = "rsi_15m" if "15" in tf else "rsi_1h"
        rsi = enhanced_ta.get(key) or price_data.get("rsi", 50)
        if rsi is None:
            return False, "RSI unavailable"
        result = _cmp(rsi, op, val)
        return result, f"RSI({tf})={rsi:.1f} {op} {val}"

    # MACD crossover / direction
    if name == "macd":
        macd = enhanced_ta.get("macd", {})
        if not macd:
            return False, "MACD unavailable"
        crossover = macd.get("crossover", "")
        sub = cond.get("condition", op)
        if sub in ("bullish", "bullish_cross", "golden"):
            result = crossover in ("BULLISH", "BULLISH_CROSS")
        elif sub in ("bearish", "bearish_cross", "death"):
            result = crossover in ("BEARISH", "BEARISH_CROSS")
        elif sub == "crosses_above":
            result = crossover == "BULLISH_CROSS"
        elif sub == "crosses_below":
            result = crossover == "BEARISH_CROSS"
        else:
            hist = macd.get("histogram", 0)
            result = _cmp(hist, op, val)
        return result, f"MACD={crossover}"

    # EMA crossover
    if name in ("ema", "ema_cross"):
        ema = enhanced_ta.get("ema_cross", {})
        if not ema:
            return False, "EMA unavailable"
        sig  = ema.get("signal", "")
        sub  = cond.get("condition", op)
        if sub in ("bullish",):
            result = sig in ("BULLISH", "GOLDEN_CROSS")
        elif sub in ("bearish",):
            result = sig in ("BEARISH", "DEATH_CROSS")
        elif sub == "golden_cross":
            result = sig == "GOLDEN_CROSS"
        elif sub == "death_cross":
            result = sig == "DEATH_CROSS"
        else:
            spread = ema.get("spread_pct", 0)
            result = _cmp(spread, op, val)
        return result, f"EMA={sig}"

    # Bollinger Bands
    if name in ("bb", "bollinger"):
        bb = enhanced_ta.get("bollinger", {})
        if not bb:
            return False, "BB unavailable"
        sub = cond.get("condition", "")
        if sub == "squeeze":
            result = bb.get("squeeze") in ("SQUEEZE", "TIGHT")
        elif sub == "above_upper":
            result = bb.get("percent_b", 50) > 100
        elif sub == "below_lower":
            result = bb.get("percent_b", 50) < 0
        elif sub == "overbought":
            result = bb.get("percent_b", 50) >= val
        elif sub == "oversold":
            result = bb.get("percent_b", 50) <= val
        else:
            pb = bb.get("percent_b", 50)
            result = _cmp(pb, op, val)
        return result, f"BB %B={bb.get('percent_b', 0):.0f}"

    # VWAP deviation
    if name == "vwap":
        vwap = enhanced_ta.get("vwap", {})
        if not vwap:
            return False, "VWAP unavailable"
        dev = vwap.get("deviation_pct", 0)
        result = _cmp(dev, op, val)
        return result, f"VWAP dev={dev:+.2f}%"

    # Volume ratio
    if name in ("volume", "volume_ratio"):
        vr = price_data.get("volume_ratio", 1.0)
        result = _cmp(vr, op, val)
        return result, f"Vol ratio={vr:.2f}x"

    return False, f"Unknown indicator: {name}"


# ─────────────────────────────────────────────────────────────────────────────
# Price momentum  (X% move in Y minutes)
# ─────────────────────────────────────────────────────────────────────────────

async def eval_price_momentum(cond: Dict, symbol: str, http_client) -> Tuple[bool, str]:
    """
    price_momentum: did price move X% in the last window_minutes?
    cond keys: window_minutes, operator, value, direction (up|down|any)
    """
    window   = int(cond.get("window_minutes", 10))
    op       = cond.get("operator", "gt")
    val      = float(cond.get("value", 5))
    req_dir  = cond.get("direction", "any")

    # Pick candle interval — 1m for short windows, 5m for longer
    interval = "1m" if window <= 15 else "5m"
    limit    = max(window, 15)

    try:
        url  = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}"
        resp = await http_client.get(url, timeout=6)
        if resp.status_code != 200:
            return False, "Kline fetch failed"
        klines = resp.json()
        if len(klines) < 2:
            return False, "Not enough candles"

        open_price  = float(klines[0][1])
        close_price = float(klines[-1][4])
        pct_change  = (close_price - open_price) / open_price * 100

        if req_dir == "up" and pct_change < 0:
            return False, f"Price moved {pct_change:+.2f}% (need up)"
        if req_dir == "down" and pct_change > 0:
            return False, f"Price moved {pct_change:+.2f}% (need down)"

        result = _cmp(abs(pct_change), op, val)
        return result, f"Price Δ({window}min)={pct_change:+.2f}%"
    except Exception as e:
        return False, f"Momentum error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Volume spike
# ─────────────────────────────────────────────────────────────────────────────

def eval_volume_spike(cond: Dict, price_data: Dict) -> Tuple[bool, str]:
    """Volume is X times its 20-period average."""
    multiplier = float(cond.get("multiplier", 1.5))
    vr         = price_data.get("volume_ratio", 1.0)
    result     = vr >= multiplier
    return result, f"Vol spike={vr:.2f}x (need {multiplier}x)"


# ─────────────────────────────────────────────────────────────────────────────
# Support / Resistance
# ─────────────────────────────────────────────────────────────────────────────

def eval_support_resistance(cond: Dict, enhanced_ta: Dict, current_price: float) -> Tuple[bool, str]:
    """
    condition: at_support | at_resistance | breakout_above | breakout_below
    tolerance_pct: how close to level counts as "at" (default 1%)
    """
    sub       = cond.get("condition", "at_support")
    tolerance = float(cond.get("tolerance_pct", 1.0)) / 100
    sr        = enhanced_ta.get("support_resistance", {})
    if not sr:
        return False, "S/R unavailable"

    supports    = sr.get("supports", [])
    resistances = sr.get("resistances", [])

    if sub == "at_support":
        for s in supports:
            if abs(current_price - s) / s <= tolerance:
                return True, f"Price at support {s:.6f} (±{tolerance*100:.1f}%)"
        return False, "Not at support"

    if sub == "at_resistance":
        for r in resistances:
            if abs(current_price - r) / r <= tolerance:
                return True, f"Price at resistance {r:.6f}"
        return False, "Not at resistance"

    if sub == "breakout_above":
        for r in resistances:
            if current_price > r * (1 + tolerance * 0.5):
                return True, f"Broke above resistance {r:.6f}"
        return False, "No resistance breakout"

    if sub == "breakout_below":
        for s in supports:
            if current_price < s * (1 - tolerance * 0.5):
                return True, f"Broke below support {s:.6f}"
        return False, "No support breakdown"

    return False, f"Unknown S/R condition: {sub}"


# ─────────────────────────────────────────────────────────────────────────────
# Fair Value Gap (FVG)
# ─────────────────────────────────────────────────────────────────────────────

def detect_fvg(klines: List) -> List[Dict]:
    """
    Detect Fair Value Gaps in a kline list.
    A bullish FVG: candle[i-1].high < candle[i+1].low  (gap between c1 high and c3 low)
    A bearish FVG: candle[i-1].low  > candle[i+1].high
    Returns list of {type, top, bottom, mid, candle_index}
    """
    gaps = []
    for i in range(1, len(klines) - 1):
        c1_high = float(klines[i - 1][2])
        c1_low  = float(klines[i - 1][3])
        c3_high = float(klines[i + 1][2])
        c3_low  = float(klines[i + 1][3])

        if c1_high < c3_low:
            # Bullish FVG — gap above c1 high and below c3 low
            gaps.append({
                "type":   "bullish",
                "top":    c3_low,
                "bottom": c1_high,
                "mid":    (c1_high + c3_low) / 2,
                "idx":    i,
            })
        elif c1_low > c3_high:
            # Bearish FVG — gap below c1 low and above c3 high
            gaps.append({
                "type":   "bearish",
                "top":    c1_low,
                "bottom": c3_high,
                "mid":    (c3_high + c1_low) / 2,
                "idx":    i,
            })

    return gaps


async def eval_fvg(cond: Dict, symbol: str, current_price: float, http_client) -> Tuple[bool, str]:
    """
    FVG conditions: price_in_gap | gap_exists | gap_filled
    direction: bullish | bearish | any
    timeframe: 5m | 15m | 1h
    lookback: how many candles back to look for gaps (default 20)
    """
    sub       = cond.get("condition", "price_in_gap")
    direction = cond.get("direction", "any")
    tf        = cond.get("timeframe", "15m")
    lookback  = int(cond.get("lookback", 20)) + 2

    try:
        url   = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={tf}&limit={lookback}"
        resp  = await http_client.get(url, timeout=6)
        if resp.status_code != 200:
            return False, "FVG kline fetch failed"
        klines = resp.json()
        if len(klines) < 3:
            return False, "Not enough candles for FVG"

        gaps = detect_fvg(klines)
        if not gaps:
            return False, "No FVG detected"

        # Filter by direction
        if direction != "any":
            gaps = [g for g in gaps if g["type"] == direction]
        if not gaps:
            return False, f"No {direction} FVG detected"

        # Take the most recent gap
        gap = gaps[-1]

        if sub == "gap_exists":
            return True, f"{gap['type'].title()} FVG at {gap['bottom']:.6f}–{gap['top']:.6f}"

        if sub == "price_in_gap":
            in_gap = gap["bottom"] <= current_price <= gap["top"]
            return in_gap, (
                f"Price {'IN' if in_gap else 'NOT in'} {gap['type']} FVG "
                f"({gap['bottom']:.6f}–{gap['top']:.6f})"
            )

        if sub == "gap_filled":
            if gap["type"] == "bullish":
                filled = current_price <= gap["bottom"]
            else:
                filled = current_price >= gap["top"]
            return filled, f"FVG {'filled' if filled else 'unfilled'}"

        return False, f"Unknown FVG condition: {sub}"

    except Exception as e:
        return False, f"FVG error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Master evaluator — runs all conditions in a strategy config
# ─────────────────────────────────────────────────────────────────────────────

async def evaluate_strategy_conditions(
    strategy_config: Dict,
    symbol: str,
    price_data: Dict,
    enhanced_ta: Dict,
    http_client,
) -> Tuple[bool, List[str]]:
    """
    Evaluate all entry_conditions in a strategy config.
    Returns (all_passed: bool, detail_lines: list[str])
    """
    entry  = strategy_config.get("entry_conditions", {})
    op     = entry.get("operator", "AND").upper()
    conds  = entry.get("conditions", [])
    current_price = price_data.get("price", 0)

    results  = []
    details  = []

    for cond in conds:
        ctype = cond.get("type", "")

        if ctype == "indicator":
            passed, detail = eval_indicator(cond, price_data, enhanced_ta)
        elif ctype == "price_momentum":
            passed, detail = await eval_price_momentum(cond, symbol, http_client)
        elif ctype == "volume_spike":
            passed, detail = eval_volume_spike(cond, price_data)
        elif ctype == "support_resistance":
            passed, detail = eval_support_resistance(cond, enhanced_ta, current_price)
        elif ctype == "fvg":
            passed, detail = await eval_fvg(cond, symbol, current_price, http_client)
        else:
            passed, detail = False, f"Unknown condition type: {ctype}"

        results.append(passed)
        details.append(f"{'✅' if passed else '❌'} {detail}")

    if not results:
        return False, ["No conditions defined"]

    if op == "AND":
        return all(results), details
    else:  # OR
        return any(results), details
