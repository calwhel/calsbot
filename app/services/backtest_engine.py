"""
Backtest Engine — TradeHub Strategy Portal

Replays a wizard strategy config against historical OHLCV data.
Uses Binance Futures (falling back to Binance spot) for candle history.
All indicator math is synchronous and self-contained; no live API calls during replay.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# ── Candle field helpers ────────────────────────────────────────────────────────
def _closes(k): return [float(x[4]) for x in k]
def _highs(k):  return [float(x[2]) for x in k]
def _lows(k):   return [float(x[3]) for x in k]
def _opens(k):  return [float(x[1]) for x in k]
def _vols(k):   return [float(x[5]) for x in k]

# ── Indicator math (synchronous, no HTTP) ───────────────────────────────────────
def _ema_list(data: List[float], period: int) -> List[float]:
    if not data: return []
    k = 2 / (period + 1)
    ema = [data[0]]
    for v in data[1:]:
        ema.append(v * k + ema[-1] * (1 - k))
    return ema

def _ema(data: List[float], period: int) -> Optional[float]:
    if len(data) < period: return None
    return _ema_list(data, period)[-1]

def _sma(data: List[float], period: int) -> Optional[float]:
    if len(data) < period: return None
    return sum(data[-period:]) / period

def _rsi_values(closes: List[float], period: int = 14) -> List[float]:
    if len(closes) < period + 1:
        return []
    gains, losses = [], []
    for i in range(1, len(closes)):
        d = closes[i] - closes[i - 1]
        gains.append(max(d, 0.0))
        losses.append(max(-d, 0.0))
    ag = sum(gains[:period]) / period
    al = sum(losses[:period]) / period
    rsi = []
    for i in range(period, len(closes)):
        if i > period:
            ag = (ag * (period - 1) + gains[i - 1]) / period
            al = (al * (period - 1) + losses[i - 1]) / period
        rsi.append(100 - 100 / (1 + ag / (al or 1e-10)))
    return rsi

def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    v = _rsi_values(closes, period)
    return v[-1] if v else None

def _macd(closes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
    if len(closes) < slow + signal:
        return None
    fast_ema = _ema_list(closes, fast)
    slow_ema = _ema_list(closes, slow)
    macd_line = [f - s for f, s in zip(fast_ema[slow - 1:], slow_ema[slow - 1:])]
    if len(macd_line) < signal:
        return None
    sig_line = _ema_list(macd_line, signal)
    histogram = macd_line[-1] - sig_line[-1]
    prev_hist = (macd_line[-2] - sig_line[-2]) if len(macd_line) > signal else None
    cross = "NONE"
    if prev_hist is not None:
        if prev_hist < 0 and histogram > 0:
            cross = "BULLISH_CROSS"
        elif prev_hist > 0 and histogram < 0:
            cross = "BEARISH_CROSS"
        elif histogram > 0:
            cross = "BULLISH"
        else:
            cross = "BEARISH"
    return {"histogram": histogram, "cross": cross}

def _bb(closes: List[float], period: int = 20, std_mult: float = 2.0) -> Optional[Dict]:
    if len(closes) < period:
        return None
    w = closes[-period:]
    mid = sum(w) / period
    variance = sum((x - mid) ** 2 for x in w) / period
    std = variance ** 0.5
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    width = (upper - lower) / mid * 100 if mid else 0
    return {"upper": upper, "lower": lower, "mid": mid, "width": width, "squeeze": width < 3.0}

def _stochrsi(closes: List[float], rsi_period: int = 14, stoch_period: int = 14) -> Optional[Dict]:
    rsi_vals = _rsi_values(closes, rsi_period)
    if len(rsi_vals) < stoch_period:
        return None
    window = rsi_vals[-stoch_period:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return {"k": 50.0}
    k = (rsi_vals[-1] - lo) / (hi - lo) * 100
    prev_k = (rsi_vals[-2] - lo) / (hi - lo) * 100 if len(rsi_vals) > stoch_period else k
    return {"k": k, "prev_k": prev_k}

def _supertrend(klines: List, period: int = 10, multiplier: float = 3.0) -> str:
    if len(klines) < period + 1:
        return "UNKNOWN"
    tr_list = []
    for i in range(1, len(klines)):
        h, l = float(klines[i][2]), float(klines[i][3])
        pc = float(klines[i - 1][4])
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr_vals = _ema_list(tr_list, period)
    atr = atr_vals[-1]
    hl2 = (float(klines[-1][2]) + float(klines[-1][3])) / 2
    basic_lower = hl2 - multiplier * atr
    curr_close = float(klines[-1][4])
    return "BULLISH" if curr_close > basic_lower else "BEARISH"

def _vol_ratio(klines: List, lookback: int = 20) -> float:
    vols = _vols(klines)
    if len(vols) < lookback + 1:
        return 1.0
    avg = sum(vols[-lookback - 1:-1]) / lookback
    return vols[-1] / avg if avg > 0 else 1.0

def _price_momentum_pct(klines: List, window_candles: int) -> Optional[float]:
    if len(klines) < window_candles + 1:
        return None
    ref_open = float(klines[-window_candles][1])
    curr_close = float(klines[-1][4])
    if ref_open == 0:
        return None
    return (curr_close - ref_open) / ref_open * 100

def _cmp(actual: float, operator: str, threshold: float) -> bool:
    return {
        "gt":  actual > threshold,
        "gte": actual >= threshold,
        "lt":  actual < threshold,
        "lte": actual <= threshold,
        "eq":  abs(actual - threshold) < 0.001,
    }.get(operator, False)

def _atr_values(klines: List, period: int = 14) -> List[float]:
    """True Range list then Wilder-smoothed ATR values."""
    if len(klines) < 2:
        return []
    trs = []
    for i in range(1, len(klines)):
        h  = float(klines[i][2]); l  = float(klines[i][3])
        pc = float(klines[i - 1][4])
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return []
    # Wilder smoothing (same as Wilder RSI)
    atr = sum(trs[:period]) / period
    result = [atr]
    for tr in trs[period:]:
        atr = (atr * (period - 1) + tr) / period
        result.append(atr)
    return result

def _atr(klines: List, period: int = 14) -> Optional[float]:
    v = _atr_values(klines, period)
    return v[-1] if v else None

def _vwap(klines: List) -> Optional[float]:
    """Simple VWAP over all available candles."""
    total_vol = sum(float(k[5]) for k in klines)
    if total_vol == 0:
        return None
    total_pv = sum(
        ((float(k[2]) + float(k[3]) + float(k[4])) / 3) * float(k[5])
        for k in klines
    )
    return total_pv / total_vol

def _williams_r(klines: List, period: int = 14) -> Optional[float]:
    """Williams %R: range -100 to 0.  Oversold < -80, overbought > -20."""
    if len(klines) < period:
        return None
    window = klines[-period:]
    hh = max(float(k[2]) for k in window)
    ll = min(float(k[3]) for k in window)
    close = float(klines[-1][4])
    if hh == ll:
        return -50.0
    return (hh - close) / (hh - ll) * -100.0

def _adx(klines: List, period: int = 14) -> Optional[float]:
    """Average Directional Index.  >25 = trending, <25 = ranging."""
    if len(klines) < period * 2 + 1:
        return None
    pos_dms, neg_dms, trs = [], [], []
    for i in range(1, len(klines)):
        h  = float(klines[i][2]);  l  = float(klines[i][3])
        ph = float(klines[i-1][2]); pl = float(klines[i-1][3])
        pc = float(klines[i-1][4])
        up_move   = h - ph
        down_move = pl - l
        pos_dms.append(up_move   if up_move   > down_move and up_move   > 0 else 0.0)
        neg_dms.append(down_move if down_move > up_move   and down_move > 0 else 0.0)
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    if len(trs) < period:
        return None
    # First Wilder smoothing
    atr_s  = sum(trs[:period])
    pdm_s  = sum(pos_dms[:period])
    ndm_s  = sum(neg_dms[:period])
    dx_list = []
    for i in range(period, len(trs)):
        atr_s  = atr_s  - atr_s  / period + trs[i]
        pdm_s  = pdm_s  - pdm_s  / period + pos_dms[i]
        ndm_s  = ndm_s  - ndm_s  / period + neg_dms[i]
        pdi = 100 * pdm_s / atr_s if atr_s else 0
        ndi = 100 * ndm_s / atr_s if atr_s else 0
        denom = pdi + ndi
        dx_list.append(100 * abs(pdi - ndi) / denom if denom else 0)
    if len(dx_list) < period:
        return None
    adx = sum(dx_list[:period]) / period
    for dx in dx_list[period:]:
        adx = (adx * (period - 1) + dx) / period
    return adx

def _keltner(klines: List, ema_period: int = 20, atr_mult: float = 2.0) -> Optional[Dict]:
    """Keltner Channel: mid=EMA(close), upper/lower=mid ± mult*ATR."""
    closes = _closes(klines)
    mid    = _ema(closes, ema_period)
    atr    = _atr(klines, 14)
    if mid is None or atr is None:
        return None
    return {"upper": mid + atr_mult * atr, "lower": mid - atr_mult * atr, "mid": mid}

def _pivot_range(klines: List, lookback: int = 20) -> Optional[Dict]:
    """High/low of the prior `lookback` candles (excluding the last one)."""
    if len(klines) < lookback + 1:
        return None
    window = klines[-(lookback + 1):-1]
    return {
        "high": max(float(k[2]) for k in window),
        "low":  min(float(k[3]) for k in window),
    }

def _swing_highs(values: List[float], wing: int = 3) -> List[int]:
    """Return indices of swing highs (local maxima) in `values`."""
    out = []
    for i in range(wing, len(values) - wing):
        if all(values[i] >= values[i - j] for j in range(1, wing + 1)) and \
           all(values[i] >= values[i + j] for j in range(1, wing + 1)):
            out.append(i)
    return out

def _swing_lows(values: List[float], wing: int = 3) -> List[int]:
    """Return indices of swing lows (local minima) in `values`."""
    out = []
    for i in range(wing, len(values) - wing):
        if all(values[i] <= values[i - j] for j in range(1, wing + 1)) and \
           all(values[i] <= values[i + j] for j in range(1, wing + 1)):
            out.append(i)
    return out

def _detect_divergence(klines: List, indicator: str = "rsi", direction: str = "bullish") -> bool:
    """
    Detect classic divergence between price and an oscillator.

    Bullish divergence  — price makes a lower low, oscillator makes a higher low.
    Bearish divergence  — price makes a higher high, oscillator makes a lower high.

    Uses the last 60 candles and a wing of 3 candles each side to identify swings.
    """
    lookback = min(len(klines), 60)
    k = klines[-lookback:]
    closes = _closes(k)

    # Build oscillator series — must be exactly len(closes) long (pad front with neutral)
    if indicator == "macd":
        m_series: List[float] = []
        for end in range(1, len(closes) + 1):
            m = _macd(closes[:end])
            m_series.append(m["histogram"] if m else 0.0)
        osc = m_series
    else:  # default: rsi
        rsi_all = _rsi_values(closes, 14)
        osc = [50.0] * (len(closes) - len(rsi_all)) + rsi_all

    if len(osc) != len(closes):
        return False

    if direction == "bullish":
        # Need two swing lows in closes (use actual lows for price)
        price_series = [float(c[3]) for c in k]  # lows
        idxs = _swing_lows(price_series, wing=3)
        if len(idxs) < 2:
            return False
        i1, i2 = idxs[-2], idxs[-1]
        # Price makes lower low, oscillator makes higher low
        return price_series[i2] < price_series[i1] and osc[i2] > osc[i1]
    else:
        # Bearish: price higher high, oscillator lower high
        price_series = [float(c[2]) for c in k]  # highs
        idxs = _swing_highs(price_series, wing=3)
        if len(idxs) < 2:
            return False
        i1, i2 = idxs[-2], idxs[-1]
        return price_series[i2] > price_series[i1] and osc[i2] < osc[i1]

def _detect_fibonacci(klines: List, level_str: str = "0.618", fib_type: str = "at_retracement",
                      tol_pct: float = 1.0) -> bool:
    """
    Detect whether the current price is at a Fibonacci retracement/extension level.

    Finds the most recent significant swing high and swing low over the last 100
    candles, computes the fib grid, and checks if price is within tol_pct%.
    """
    lookback = min(len(klines), 100)
    k = klines[-lookback:]
    highs = [float(c[2]) for c in k]
    lows  = [float(c[3]) for c in k]
    swing_h_idxs = _swing_highs(highs, wing=5)
    swing_l_idxs = _swing_lows(lows,  wing=5)
    if not swing_h_idxs or not swing_l_idxs:
        return False

    sh = highs[swing_h_idxs[-1]]
    sl = lows[swing_l_idxs[-1]]
    if sh == sl:
        return False

    try:
        level = float(level_str)
    except ValueError:
        return False

    rng   = sh - sl
    close = float(klines[-1][4])

    # Retracement levels sit between swing_low and swing_high
    target = sl + (1 - level) * rng if fib_type == "at_retracement" else sh + level * rng
    return abs(close - target) / target * 100 <= tol_pct

def _detect_fvg(klines: List, fvg_dir: str = "bullish", min_gap_pct: float = 0.3) -> bool:
    """
    Fair Value Gap (FVG / Imbalance).

    Scans the last 50 candles for any historical FVG zone, then checks if the
    current close is retesting (trading inside) that zone.

    Bullish FVG — three-candle sequence where candle[A].high < candle[A+2].low.
                  Gap zone = [candle[A].high, candle[A+2].low].
                  Signal fires when current price trades back inside that zone.

    Bearish FVG — candle[A].low > candle[A+2].high.
                  Gap zone = [candle[A+2].high, candle[A].low].
    """
    if len(klines) < 5:
        return False
    close   = float(klines[-1][4])
    lookback = min(len(klines) - 1, 50)

    for i in range(2, lookback):
        c_a  = klines[-(i + 2)]   # older candle (A)
        c_c  = klines[-i]         # newer candle (A+2)

        if fvg_dir == "bullish":
            gap_lo = float(c_a[2])   # candle A high
            gap_hi = float(c_c[3])   # candle A+2 low
            if gap_hi > gap_lo:
                gap_pct = (gap_hi - gap_lo) / gap_lo * 100
                if gap_pct >= min_gap_pct and gap_lo <= close <= gap_hi:
                    return True
        else:
            gap_hi = float(c_a[3])   # candle A low
            gap_lo = float(c_c[2])   # candle A+2 high
            if gap_lo < gap_hi:
                gap_pct = (gap_hi - gap_lo) / gap_lo * 100
                if gap_pct >= min_gap_pct and gap_lo <= close <= gap_hi:
                    return True
    return False

def _detect_order_block(klines: List, ob_type: str = "bullish") -> bool:
    """
    Order Block detection (SMC).

    Bullish OB — the last *bearish* (red) candle before a strong bullish impulse.
               Price must currently be retesting (trading inside) that candle's body.
    Bearish OB — the last *bullish* (green) candle before a strong bearish impulse.
               Price must currently be retesting that candle's body.

    A "strong impulse" = the next 3 candles move at least 2× the OB candle's body.
    """
    if len(klines) < 10:
        return False
    close = float(klines[-1][4])
    # Scan last 50 candles (excluding current)
    lookback = min(len(klines) - 1, 50)
    for i in range(len(klines) - 2, len(klines) - 2 - lookback, -1):
        c  = klines[i]
        o  = float(c[1]); cl = float(c[4])
        body = abs(cl - o)
        if body == 0:
            continue
        is_bearish = cl < o
        is_bullish = cl > o
        if ob_type == "bullish" and not is_bearish:
            continue
        if ob_type == "bearish" and not is_bullish:
            continue
        # Check impulse: next 3 candles move at least 2× OB body
        impulse_candles = klines[i + 1: i + 4]
        if len(impulse_candles) < 3:
            continue
        if ob_type == "bullish":
            move = float(impulse_candles[-1][4]) - float(impulse_candles[0][1])
            if move < body * 2:
                continue
            # Price retesting: inside OB body range [min(o,cl), max(o,cl)]
            ob_lo, ob_hi = min(o, cl), max(o, cl)
        else:
            move = float(impulse_candles[0][1]) - float(impulse_candles[-1][4])
            if move < body * 2:
                continue
            ob_lo, ob_hi = min(o, cl), max(o, cl)
        if ob_lo <= close <= ob_hi:
            return True
    return False

def _detect_market_structure(klines: List, condition: str = "bos_bullish") -> bool:
    """
    Market Structure — Break of Structure (BOS) and Change of Character (CHoCH).

    BOS bullish  — current close breaks above the last significant swing high.
    BOS bearish  — current close breaks below the last significant swing low.
    CHoCH bullish — after a downtrend (series of lower highs), close breaks above last lower high.
    CHoCH bearish — after an uptrend (series of higher lows), close breaks below last higher low.
    """
    lookback = min(len(klines), 80)
    k      = klines[-lookback:]
    highs  = [float(c[2]) for c in k]
    lows   = [float(c[3]) for c in k]
    close  = float(klines[-1][4])

    sh_idxs = _swing_highs(highs, wing=4)
    sl_idxs = _swing_lows(lows,  wing=4)

    if condition == "bos_bullish":
        if not sh_idxs: return False
        return close > highs[sh_idxs[-1]]

    if condition == "bos_bearish":
        if not sl_idxs: return False
        return close < lows[sl_idxs[-1]]

    if condition == "choch_bullish":
        # At least 2 swing highs; last swing high lower than the one before it (downtrend)
        if len(sh_idxs) < 2: return False
        lower_high = highs[sh_idxs[-1]] < highs[sh_idxs[-2]]
        return lower_high and close > highs[sh_idxs[-1]]

    if condition == "choch_bearish":
        if len(sl_idxs) < 2: return False
        higher_low = lows[sl_idxs[-1]] > lows[sl_idxs[-2]]
        return higher_low and close < lows[sl_idxs[-1]]

    return False

def _detect_ifvg(klines: List, direction: str = "bullish", min_gap_pct: float = 0.3) -> bool:
    """
    Inverted Fair Value Gap (IFVG).

    A previously filled FVG that has flipped to act as support (bullish IFVG)
    or resistance (bearish IFVG).  Detection steps:
      1. Scan for any historical FVG gap in the last 50 candles.
      2. Confirm the gap was filled (price traded through it at least once).
      3. Check if current close is retesting the flipped zone from the other side.
    """
    if len(klines) < 8:
        return False
    close = float(klines[-1][4])
    lookback = min(len(klines) - 3, 50)

    for i in range(3, lookback):
        c_a = klines[-(i + 2)]
        c_c = klines[-i]

        if direction == "bullish":
            gap_lo = float(c_a[2])   # candle A high
            gap_hi = float(c_c[3])   # candle A+2 low
            if gap_hi <= gap_lo:
                continue
            gap_pct = (gap_hi - gap_lo) / gap_lo * 100
            if gap_pct < min_gap_pct:
                continue
            # Check if gap was filled: any candle in between traded below gap_lo
            filled = any(float(klines[-(j)][3]) < gap_lo for j in range(2, i))
            if not filled:
                continue
            # Now check retest: price is near or inside the zone from below (support test)
            if gap_lo * 0.998 <= close <= gap_hi * 1.002:
                return True

        else:  # bearish IFVG
            gap_hi = float(c_a[3])   # candle A low
            gap_lo = float(c_c[2])   # candle A+2 high
            if gap_lo >= gap_hi:
                continue
            gap_pct = (gap_hi - gap_lo) / gap_lo * 100
            if gap_pct < min_gap_pct:
                continue
            filled = any(float(klines[-(j)][2]) > gap_hi for j in range(2, i))
            if not filled:
                continue
            if gap_lo * 0.998 <= close <= gap_hi * 1.002:
                return True
    return False


def _detect_breaker_block(klines: List, direction: str = "bullish") -> bool:
    """
    Breaker Block — a failed order block that has been violated and retested.

    Bullish breaker: find a bearish OB that price broke below (violated), then
    price retraces back up to retest the OB zone from below.
    Bearish breaker: find a bullish OB that price broke above (violated), then
    price retraces back down to retest from above.
    """
    if len(klines) < 12:
        return False
    close = float(klines[-1][4])
    lookback = min(len(klines) - 4, 50)

    for i in range(4, lookback):
        c  = klines[-(i + 2)]
        o  = float(c[1]); cl = float(c[4])
        body = abs(cl - o)
        if body == 0:
            continue
        ob_lo, ob_hi = min(o, cl), max(o, cl)
        is_bearish_candle = cl < o
        is_bullish_candle = cl > o

        if direction == "bullish" and is_bearish_candle:
            # OB was a bearish candle — check impulse up after it
            impulse = klines[-(i + 1): -(i - 2) if i > 2 else None]
            if len(impulse) < 2:
                continue
            move_up = float(impulse[-1][4]) - float(impulse[0][1])
            if move_up < body * 2:
                continue
            # Check that price subsequently broke below ob_lo (violated the OB)
            subsequent = klines[-(i - 1):]
            if not any(float(k[3]) < ob_lo for k in subsequent[:-1]):
                continue
            # Current price retesting from below (now acting as resistance-turned-support)
            if ob_lo * 0.998 <= close <= ob_hi * 1.002:
                return True

        elif direction == "bearish" and is_bullish_candle:
            # OB was a bullish candle — check impulse down after it
            impulse = klines[-(i + 1): -(i - 2) if i > 2 else None]
            if len(impulse) < 2:
                continue
            move_down = float(impulse[0][1]) - float(impulse[-1][4])
            if move_down < body * 2:
                continue
            subsequent = klines[-(i - 1):]
            if not any(float(k[2]) > ob_hi for k in subsequent[:-1]):
                continue
            if ob_lo * 0.998 <= close <= ob_hi * 1.002:
                return True
    return False


def _detect_mss(klines: List, direction: str = "bullish", lookback: int = 20) -> bool:
    """
    Market Structure Shift (MSS).

    Bullish MSS: current close breaks above the most recent swing high
                 within the last `lookback` candles.
    Bearish MSS: current close breaks below the most recent swing low.

    Uses a tighter lookback (default 20) than the general market_structure
    evaluator (80) to catch earlier, more localised structure breaks.
    """
    n = min(len(klines), lookback + 4)
    k = klines[-n:]
    highs = [float(c[2]) for c in k]
    lows  = [float(c[3]) for c in k]
    close = float(klines[-1][4])

    sh_idxs = _swing_highs(highs, wing=3)
    sl_idxs = _swing_lows(lows,  wing=3)

    if direction == "bullish":
        if not sh_idxs:
            return False
        return close > highs[sh_idxs[-1]]

    if direction == "bearish":
        if not sl_idxs:
            return False
        return close < lows[sl_idxs[-1]]

    return False


def _detect_choch(klines: List, direction: str = "bullish") -> bool:
    """
    Change of Character (CHoCH) — the first structure break after a series of
    lower highs (bearish CHoCH → bullish reversal) or higher lows (bullish CHoCH
    → bearish reversal).

    Slightly tighter wing (2 candles each side) to catch earlier signal than BOS.
    """
    lookback = min(len(klines), 60)
    k = klines[-lookback:]
    highs = [float(c[2]) for c in k]
    lows  = [float(c[3]) for c in k]
    close = float(klines[-1][4])

    sh_idxs = _swing_highs(highs, wing=2)
    sl_idxs = _swing_lows(lows,  wing=2)

    if direction == "bullish":
        if len(sh_idxs) < 2:
            return False
        lower_high = highs[sh_idxs[-1]] < highs[sh_idxs[-2]]
        return lower_high and close > highs[sh_idxs[-1]]

    if direction == "bearish":
        if len(sl_idxs) < 2:
            return False
        higher_low = lows[sl_idxs[-1]] > lows[sl_idxs[-2]]
        return higher_low and close < lows[sl_idxs[-1]]

    return False


def _detect_liquidity_sweep(klines: List, direction: str = "bullish",
                             sweep_lookback: int = 10) -> bool:
    """
    Liquidity Sweep — price wicks beyond a recent swing high/low then closes back inside.

    Bullish LQ: wick sweeps below a recent swing low (stop hunt), then current
                candle closes back above that low.
    Bearish LQ: wick sweeps above a recent swing high, then closes back below.

    The sweep is detected on the most recent candle.
    """
    if len(klines) < sweep_lookback + 2:
        return False

    window = klines[-(sweep_lookback + 1):-1]   # prior N candles, excluding current
    cur    = klines[-1]
    cur_open  = float(cur[1])
    cur_high  = float(cur[2])
    cur_low   = float(cur[3])
    cur_close = float(cur[4])

    if direction == "bullish":
        swing_low = min(float(k[3]) for k in window)
        swept  = cur_low < swing_low      # wick dipped below swing low
        closed = cur_close > swing_low    # but closed back above
        return swept and closed

    if direction == "bearish":
        swing_high = max(float(k[2]) for k in window)
        swept  = cur_high > swing_high
        closed = cur_close < swing_high
        return swept and closed

    return False


def _detect_mitigation_block(klines: List, direction: str = "bullish",
                              min_move_mult: float = 2.5) -> bool:
    """
    Mitigation Block — price retraces to the 50% level of the origin candle
    that launched a strong impulsive move.

    Bullish MIT: find a strong bullish impulse candle, compute its 50% level,
                 signal when current price retraces to that 50%.
    Bearish MIT: find a strong bearish impulse candle and retest its 50%.
    """
    if len(klines) < 10:
        return False

    close    = float(klines[-1][4])
    lookback = min(len(klines) - 1, 50)

    # Compute average candle size over the window
    bodies   = [abs(float(k[4]) - float(k[1])) for k in klines[-lookback:]]
    avg_body = sum(bodies) / len(bodies) if bodies else 0
    if avg_body == 0:
        return False

    for i in range(2, lookback - 1):
        c  = klines[-(i + 1)]
        o  = float(c[1]); cl = float(c[4])
        body = abs(cl - o)
        if body < avg_body * min_move_mult:
            continue
        mid = (o + cl) / 2

        if direction == "bullish" and cl > o:
            # Strong bullish candle — price retesting its midpoint
            tol = body * 0.15
            if abs(close - mid) <= tol:
                return True

        elif direction == "bearish" and cl < o:
            tol = body * 0.15
            if abs(close - mid) <= tol:
                return True

    return False


def _detect_supply_demand_zone(klines: List, direction: str = "bullish",
                                base_lookback: int = 5, move_mult: float = 2.0) -> bool:
    """
    Supply / Demand Zone.

    Demand zone: a base (2–5 tight candles) followed by a strong bullish rally
                 (zone height × move_mult).  Price currently retesting the base.
    Supply zone: a base followed by a strong bearish drop.  Price retesting.

    `direction` = "bullish" → look for demand zone retest (long entry).
    `direction` = "bearish" → look for supply zone retest (short entry).
    """
    if len(klines) < base_lookback + 8:
        return False

    close    = float(klines[-1][4])
    lookback = min(len(klines) - base_lookback - 2, 60)

    for i in range(base_lookback + 2, lookback):
        base = klines[-(i + base_lookback): -i]
        if len(base) < 2:
            continue
        zone_hi = max(float(k[2]) for k in base)
        zone_lo = min(float(k[3]) for k in base)
        zone_sz = zone_hi - zone_lo
        if zone_sz == 0:
            continue

        # Impulse candles after the base
        impulse = klines[-i: -(i - 3) if i > 3 else None]
        if len(impulse) < 2:
            continue

        if direction == "bullish":
            move = float(impulse[-1][4]) - float(impulse[0][1])
            if move < zone_sz * move_mult:
                continue
            # Demand zone retest: price is back inside or just above the zone
            if zone_lo * 0.998 <= close <= zone_hi * 1.002:
                return True

        else:  # supply zone
            move = float(impulse[0][1]) - float(impulse[-1][4])
            if move < zone_sz * move_mult:
                continue
            if zone_lo * 0.998 <= close <= zone_hi * 1.002:
                return True

    return False


def _detect_premium_discount(klines: List, direction: str = "bullish",
                              range_lookback: int = 50) -> bool:
    """
    Premium / Discount Array.

    Calculates the range high and low of the last `range_lookback` candles.
    Midpoint (50%) divides the range into discount (below mid) and premium (above mid).

    Bullish: price is in the discount zone (<50% of range) — favour longs.
    Bearish: price is in the premium zone (>50% of range) — favour shorts.
    """
    n = min(len(klines), range_lookback)
    if n < 10:
        return False
    window    = klines[-n:]
    rng_hi    = max(float(k[2]) for k in window)
    rng_lo    = min(float(k[3]) for k in window)
    if rng_hi == rng_lo:
        return False
    mid       = (rng_hi + rng_lo) / 2
    close     = float(klines[-1][4])

    if direction == "bullish":
        return close < mid   # discount
    if direction == "bearish":
        return close > mid   # premium
    return False


def _detect_equilibrium(klines: List, direction: str = "bullish",
                         swing_lookback: int = 20, tol_pct: float = 1.5) -> bool:
    """
    Equilibrium Entry (EQ) — entry at the 50% retracement of the most recent
    significant swing, identified over the last `swing_lookback` candles.

    Bullish EQ: price pulls back to 50% of an upswing (swing_low to swing_high).
    Bearish EQ: price rallies to 50% of a downswing.
    """
    n = min(len(klines), swing_lookback + 4)
    k = klines[-n:]
    highs  = [float(c[2]) for c in k]
    lows   = [float(c[3]) for c in k]
    close  = float(klines[-1][4])

    sh_idxs = _swing_highs(highs, wing=2)
    sl_idxs = _swing_lows(lows,  wing=2)

    if not sh_idxs or not sl_idxs:
        return False

    sh = highs[sh_idxs[-1]]
    sl = lows[sl_idxs[-1]]
    eq = (sh + sl) / 2
    tol = (sh - sl) * tol_pct / 100

    if direction == "bullish":
        # Price pulled back to mid of the upswing and swing low is older than swing high
        return (sl_idxs[-1] < sh_idxs[-1]) and abs(close - eq) <= tol

    if direction == "bearish":
        return (sh_idxs[-1] < sl_idxs[-1]) and abs(close - eq) <= tol

    return False


def _detect_pin_bar(klines: List, direction: str = "bullish") -> bool:
    """
    Pin Bar — candle with small body and long rejection wick (wick ≥ 2× body).

    Bullish pin: long lower wick, small upper wick (rejection of lows).
    Bearish pin: long upper wick, small lower wick (rejection of highs).
    """
    if not klines:
        return False
    c    = klines[-1]
    o    = float(c[1]); h = float(c[2]); l = float(c[3]); cl = float(c[4])
    body = abs(cl - o)
    rng  = h - l
    if rng == 0 or body == 0:
        return False

    upper_wick = h - max(o, cl)
    lower_wick = min(o, cl) - l

    if direction == "bullish":
        return lower_wick >= body * 2 and upper_wick <= body * 0.5
    if direction == "bearish":
        return upper_wick >= body * 2 and lower_wick <= body * 0.5
    return False


def _detect_engulfing(klines: List, direction: str = "bullish") -> bool:
    """
    Engulfing Candle — current candle body fully engulfs the previous candle body.

    Bullish engulfing: closes higher, body covers the prior candle's range.
    Bearish engulfing: closes lower, body covers the prior candle's range.
    """
    if len(klines) < 2:
        return False
    o  = float(klines[-1][1]); cl = float(klines[-1][4])
    po = float(klines[-2][1]); pc = float(klines[-2][4])

    if direction == "bullish":
        return cl > o and pc < po and cl > po and o < pc
    if direction == "bearish":
        return cl < o and pc > po and cl < po and o > pc
    return False


def _detect_inside_bar(klines: List) -> bool:
    """
    Inside Bar — current candle's high and low are both within the previous
    candle's range.  Entry signal fires on the inside bar itself (the breakout
    direction is determined by the backtest direction config).
    """
    if len(klines) < 2:
        return False
    cur_hi = float(klines[-1][2]); cur_lo = float(klines[-1][3])
    prv_hi = float(klines[-2][2]); prv_lo = float(klines[-2][3])
    return cur_hi < prv_hi and cur_lo > prv_lo


def _detect_hh_hl_trend(klines: List, num_swings: int = 5) -> bool:
    """
    HH/HL Bullish Structure — series of higher highs and higher lows
    over the last `num_swings` swing points.
    """
    lookback = min(len(klines), 80)
    k = klines[-lookback:]
    highs = [float(c[2]) for c in k]
    lows  = [float(c[3]) for c in k]

    sh_idxs = _swing_highs(highs, wing=3)
    sl_idxs = _swing_lows(lows,  wing=3)

    if len(sh_idxs) < num_swings or len(sl_idxs) < num_swings:
        return False

    recent_sh = [highs[i] for i in sh_idxs[-num_swings:]]
    recent_sl = [lows[i]  for i in sl_idxs[-num_swings:]]

    hh = all(recent_sh[i] > recent_sh[i - 1] for i in range(1, len(recent_sh)))
    hl = all(recent_sl[i] > recent_sl[i - 1] for i in range(1, len(recent_sl)))
    return hh and hl


def _detect_lh_ll_trend(klines: List, num_swings: int = 5) -> bool:
    """
    LH/LL Bearish Structure — series of lower highs and lower lows.
    """
    lookback = min(len(klines), 80)
    k = klines[-lookback:]
    highs = [float(c[2]) for c in k]
    lows  = [float(c[3]) for c in k]

    sh_idxs = _swing_highs(highs, wing=3)
    sl_idxs = _swing_lows(lows,  wing=3)

    if len(sh_idxs) < num_swings or len(sl_idxs) < num_swings:
        return False

    recent_sh = [highs[i] for i in sh_idxs[-num_swings:]]
    recent_sl = [lows[i]  for i in sl_idxs[-num_swings:]]

    lh = all(recent_sh[i] < recent_sh[i - 1] for i in range(1, len(recent_sh)))
    ll = all(recent_sl[i] < recent_sl[i - 1] for i in range(1, len(recent_sl)))
    return lh and ll


def _detect_fib_retracement(klines: List, direction: str = "bullish",
                              lookback: int = 50, tol_pct: float = 1.5) -> bool:
    """
    Fibonacci Retracement entry at the 0.618 or 0.705 (OTE) zone.

    Bullish: price pulls back to 61.8–70.5% of a recent upswing.
    Bearish: price rallies to 61.8–70.5% of a recent downswing.
    """
    n = min(len(klines), lookback)
    k = klines[-n:]
    highs = [float(c[2]) for c in k]
    lows  = [float(c[3]) for c in k]

    sh_idxs = _swing_highs(highs, wing=4)
    sl_idxs = _swing_lows(lows,  wing=4)

    if not sh_idxs or not sl_idxs:
        return False

    sh = highs[sh_idxs[-1]]
    sl = lows[sl_idxs[-1]]
    if sh == sl:
        return False

    rng   = sh - sl
    close = float(klines[-1][4])

    if direction == "bullish":
        # Upswing: swing_low formed before swing_high → retracement back toward swing_low
        fib_618 = sh - 0.618 * rng
        fib_705 = sh - 0.705 * rng
        zone_lo = min(fib_618, fib_705)
        zone_hi = max(fib_618, fib_705) * (1 + tol_pct / 100)
        return zone_lo * (1 - tol_pct / 100) <= close <= zone_hi

    if direction == "bearish":
        # Downswing: retracement toward swing_high from swing_low
        fib_618 = sl + 0.618 * rng
        fib_705 = sl + 0.705 * rng
        zone_lo = min(fib_618, fib_705) * (1 - tol_pct / 100)
        zone_hi = max(fib_618, fib_705)
        return zone_lo <= close <= zone_hi * (1 + tol_pct / 100)

    return False


def _detect_vwap_session(klines: List, direction: str = "bullish") -> bool:
    """
    VWAP Session entry — price reacts to the session VWAP.

    Bullish: VWAP acts as support — price is above VWAP and the most recent
             candle's low touched or dipped near VWAP before closing back above.
    Bearish: VWAP acts as resistance — price is below VWAP and the most recent
             candle's high touched or approached VWAP before closing back below.

    VWAP is calculated from all available candles (whole session / window).
    """
    vwap = _vwap(klines)
    if vwap is None:
        return False
    cur   = klines[-1]
    c_lo  = float(cur[3])
    c_hi  = float(cur[2])
    close = float(cur[4])
    tol   = vwap * 0.003   # within 0.3% of VWAP counts as "touching"

    if direction == "bullish":
        near_vwap = c_lo <= vwap + tol
        closed_above = close > vwap
        return near_vwap and closed_above

    if direction == "bearish":
        near_vwap = c_hi >= vwap - tol
        closed_below = close < vwap
        return near_vwap and closed_below

    return False


def _session_reference(klines: List, reference: str = "session_low") -> Optional[float]:
    """
    Approximate session levels from 1h candles.
    session_low/high — min/max over last 24 candles.
    daily_open       — open of the candle that started the current UTC day.
    """
    if not klines:
        return None
    if reference == "daily_open":
        # Find the first candle of today (UTC)
        now_ts = int(klines[-1][0]) // 1000
        day_start = now_ts - (now_ts % 86400)
        for k in reversed(klines):
            if int(k[0]) // 1000 <= day_start:
                return float(k[1])
        return float(klines[0][1])
    window = klines[-min(24, len(klines)):]
    if reference == "session_low":  return min(float(k[3]) for k in window)
    if reference == "session_high": return max(float(k[2]) for k in window)
    return None

# ── Condition evaluator (no HTTP, uses pre-fetched klines) ─────────────────────
# ── Trading-session windows (UTC) — mirror strategy_executor._SESSION_HOURS ──────
# Used for replay-faithful session gating (the live evaluators use wall-clock
# time, which is wrong for a backtest; here we derive the hour from the candle
# timestamp instead). Sessions overlap by design (e.g. 14:00 UTC is in london,
# new_york AND overlap).
_BT_SESSION_HOURS = {
    "asian":    (0, 8),  "tokyo": (0, 8),  "asia": (0, 8),
    "london":   (7, 16), "europe": (7, 16),
    "new_york": (13, 22), "ny": (13, 22),
    "overlap":  (13, 16),
}

def _bt_session_active(ts_ms: int, sessions: List[str]) -> bool:
    """True if the candle timestamp's UTC hour falls in any requested session."""
    try:
        hour = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).hour
    except Exception:
        return True  # fail-open: never block a trade on a bad timestamp
    for sid in sessions:
        win = _BT_SESSION_HOURS.get(str(sid).lower().strip())
        if not win:
            continue
        a, b = win
        if a <= b:
            if a <= hour < b:
                return True
        else:  # wraps midnight (e.g. sydney)
            if hour >= a or hour < b:
                return True
    return False


# ── ICT day-trade signal ports (sync, kline-based) ───────────────────────────────
# Faithful backtest versions of the live strategy_ta.py ICT evaluators. They are
# pure price-action over the candle slice (no multi-timeframe / order-flow data),
# so unlike the no-op fx_* passthrough they are HONEST to backtest. Convention:
# klines[-1] is the closed decision bar ("current"), matching the other backtest
# evaluators (RSI/breakout read klines[-1] as the current close).

# ICT killzones (UTC) — mirror strategy_ta.eval_fx_killzone WINDOWS (hour precision).
_BT_KILLZONE_HOURS = {
    "london_kz": [(7, 9)],
    "ny_kz":     [(12, 14)],
    "asian_kz":  [(20, 23)],
}


def _bt_killzone_active(ts_ms: int, kz: str = "london_kz") -> bool:
    """True if the candle's UTC hour is inside the requested ICT killzone."""
    try:
        hour = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc).hour
    except Exception:
        return True  # fail-open
    kz = str(kz or "london_kz").lower().strip()
    if kz == "any_kz":
        windows = [w for ws in _BT_KILLZONE_HOURS.values() for w in ws]
    else:
        windows = _BT_KILLZONE_HOURS.get(kz, _BT_KILLZONE_HOURS["london_kz"])
    return any(a <= hour < b for a, b in windows)


def _bt_displacement(klines: List, direction: str = "any", min_body_ratio: float = 3.0) -> bool:
    """Institutional momentum candle: last closed body >= N × recent avg body."""
    if len(klines) < 5:
        return False
    o = _opens(klines); c = _closes(klines)
    bodies = [abs(c[i] - o[i]) for i in range(len(klines) - 1)]  # exclude current
    avg_body = sum(bodies) / len(bodies) if bodies else 0.0
    if avg_body <= 0:
        return False
    last_body = abs(c[-1] - o[-1])
    is_bull = c[-1] > o[-1]; is_bear = c[-1] < o[-1]
    dir_ok = (
        direction == "any"
        or (direction == "bullish" and is_bull)
        or (direction == "bearish" and is_bear)
    )
    return dir_ok and (last_body / avg_body) >= max(1.0, min_body_ratio)


def _bt_ote(klines: List, direction: str = "bullish", swing_lookback: int = 20,
            fib_low: float = 61.8, fib_high: float = 78.6) -> bool:
    """Optimal Trade Entry — current close inside the fib golden zone of the swing."""
    lb = max(10, int(swing_lookback))
    if len(klines) < lb + 2:
        return False
    window = klines[-(lb + 1):-1]  # recent swing, exclude current bar
    if not window:
        return False
    swing_h = max(_highs(window)); swing_l = min(_lows(window))
    rng = swing_h - swing_l
    if rng <= 0:
        return False
    flo = float(fib_low) / 100.0; fhi = float(fib_high) / 100.0
    price = _closes(klines)[-1]
    if direction == "bullish":
        zone_top = swing_h - flo * rng; zone_bot = swing_h - fhi * rng
    else:
        zone_bot = swing_l + flo * rng; zone_top = swing_l + fhi * rng
    return zone_bot <= price <= zone_top


def _bt_cisd(klines: List, direction: str = "bullish", max_run: int = 10) -> bool:
    """Change in State of Delivery — close back through the origin of the prior run."""
    try:
        max_run = max(1, min(int(max_run), 50))
    except (TypeError, ValueError):
        max_run = 10
    if len(klines) < 4:
        return False
    o = _opens(klines); c = _closes(klines)
    ci = len(c) - 1  # current closed bar = confirmation candle
    if ci < 2:
        return False
    if direction == "bullish":
        if not (c[ci] > o[ci]):
            return False
        run_start = None; j = ci - 1
        while j >= 0 and (ci - 1 - j) < max_run and c[j] < o[j]:
            run_start = j; j -= 1
        if run_start is None:
            return False
        return c[ci] > o[run_start]
    else:
        if not (c[ci] < o[ci]):
            return False
        run_start = None; j = ci - 1
        while j >= 0 and (ci - 1 - j) < max_run and c[j] > o[j]:
            run_start = j; j -= 1
        if run_start is None:
            return False
        return c[ci] < o[run_start]


def _bt_sdp(klines: List, direction: str = "bullish", swing_lookback: int = 20,
            sweep_window: int = 5, min_body_ratio: float = 2.0, max_age: int = 20) -> bool:
    """Sweep → Displacement → Pullback. Fires when current close is back inside the
    displacement FVG after a liquidity sweep (mirrors strategy_ta.eval_fx_sdp)."""
    try:
        swing_lookback = max(5, min(int(swing_lookback), 100))
        sweep_window = max(1, min(int(sweep_window), 20))
        min_body_ratio = max(1.0, float(min_body_ratio))
        max_age = max(3, min(int(max_age), 60))
    except (TypeError, ValueError):
        swing_lookback, sweep_window, min_body_ratio, max_age = 20, 5, 2.0, 20
    n = len(klines)
    if n < swing_lookback + 4:
        return False
    o = _opens(klines); h = _highs(klines); l = _lows(klines); c = _closes(klines)
    bodies = [abs(c[i] - o[i]) for i in range(n)]
    avg_body = sum(bodies) / len(bodies) if bodies else 0.0
    if avg_body <= 0:
        return False
    price = c[-1]
    start = max(swing_lookback, n - max_age)
    for d in range(n - 2, start - 1, -1):
        if d - 1 < 0 or d + 1 >= n:
            continue
        if bodies[d] < min_body_ratio * avg_body:
            continue
        if direction == "bullish" and not (c[d] > o[d]):
            continue
        if direction == "bearish" and not (c[d] < o[d]):
            continue
        if direction == "bullish":
            gap_bottom = h[d - 1]; gap_top = l[d + 1]
        else:
            gap_top = l[d - 1]; gap_bottom = h[d + 1]
        if not (gap_bottom < gap_top):
            continue
        sweep_ok = False
        for s in range(d - 1, max(d - 1 - sweep_window, 0) - 1, -1):
            ref_lo = l[max(0, s - swing_lookback):s]
            ref_hi = h[max(0, s - swing_lookback):s]
            if direction == "bullish":
                if ref_lo:
                    rm = min(ref_lo)
                    if l[s] < rm and c[s] > rm:
                        sweep_ok = True; break
            else:
                if ref_hi:
                    rm = max(ref_hi)
                    if h[s] > rm and c[s] < rm:
                        sweep_ok = True; break
        if not sweep_ok:
            continue
        if gap_bottom <= price <= gap_top:
            return True
    return False


def eval_condition_bt(cond: Dict, klines: List, interval_min: int = 5) -> bool:
    """
    Evaluate a single condition against a historical candle slice.
    Returns True for unsupported condition types (pass-through) so they
    don't falsely block signals during replay.
    """
    ctype = cond.get("type", "")

    # ── Trading session / timezone gate (timestamp-based, replay-faithful) ──────
    if ctype in ("forex_session", "session", "session_filter"):
        if not klines:
            return False
        sessions = cond.get("sessions")
        if not sessions:
            one = cond.get("session") or cond.get("name")
            sessions = [one] if one else []
        if not sessions:
            return True  # no session specified → no restriction
        return _bt_session_active(int(klines[-1][0]), sessions)

    # ── Price Momentum ──────────────────────────────────────────────────────────
    if ctype == "price_momentum":
        window_min = int(cond.get("window_minutes", 10))
        op         = cond.get("operator", "gt")
        val        = float(cond.get("value", 5))
        req_dir    = cond.get("direction", "any")
        window_c   = max(1, round(window_min / interval_min))
        pct = _price_momentum_pct(klines, window_c)
        if pct is None: return False
        if req_dir == "up"   and pct < 0: return False
        if req_dir == "down" and pct > 0: return False
        return _cmp(abs(pct), op, val)

    # ── Volume Spike ────────────────────────────────────────────────────────────
    if ctype == "volume_spike":
        mult = float(cond.get("multiplier", 1.5))
        return _vol_ratio(klines, lookback=20) >= mult

    # ── SMA direct type (remapped from wizard primaryType) ─────────────────────
    if ctype in ("sma", "sma_cross", "sma_ribbon"):
        return eval_condition_bt({**cond, "type": "indicator", "name": ctype}, klines, interval_min)

    # ── Named indicator shortcuts (confirms arrive with type = indicator name) ─
    # This handles the case where _build_confirm_cond passes through a dict whose
    # "type" key is the indicator name directly (rsi, macd, ema, etc.).
    _NAMED_INDICATORS = {"rsi", "macd", "ema", "bb", "stochrsi", "supertrend", "volume", "stoch_rsi"}
    if ctype in _NAMED_INDICATORS:
        name = "stochrsi" if ctype == "stoch_rsi" else ctype
        return eval_condition_bt({**cond, "type": "indicator", "name": name}, klines, interval_min)

    # ── Indicator ──────────────────────────────────────────────────────────────
    if ctype == "indicator":
        name    = (cond.get("name") or "").lower()
        op      = cond.get("operator", "gt")
        val     = float(cond.get("value", 50))
        sub     = cond.get("condition", "")
        closes  = _closes(klines)

        if name == "rsi":
            period = int(cond.get("period", 14))
            rsi = _rsi(closes, period)
            if rsi is None: return False
            return _cmp(rsi, op, val)

        if name == "macd":
            m = _macd(closes)
            if not m: return False
            if sub in ("bullish", "bullish_cross"):   return m["cross"] in ("BULLISH", "BULLISH_CROSS")
            if sub in ("bearish", "bearish_cross"):   return m["cross"] in ("BEARISH", "BEARISH_CROSS")
            if sub == "crosses_above":                return m["cross"] == "BULLISH_CROSS"
            if sub == "crosses_below":                return m["cross"] == "BEARISH_CROSS"
            return _cmp(m["histogram"], op, val)

        if name == "ema":
            period  = int(cond.get("period", 20))
            period2 = int(cond.get("period2", 50))
            ema_fast = _ema(closes, period)
            curr     = closes[-1]
            if sub in ("above", "price_above"):
                return ema_fast is not None and curr > ema_fast
            if sub in ("below", "price_below"):
                return ema_fast is not None and curr < ema_fast
            if sub in ("bullish_cross", "crosses_above"):
                ema_slow = _ema(closes, period2)
                pf = _ema(closes[:-1], period)  if len(closes) > period  else None
                ps = _ema(closes[:-1], period2) if len(closes) > period2 else None
                if all(v is not None for v in [ema_fast, ema_slow, pf, ps]):
                    return pf <= ps and ema_fast > ema_slow
            if sub in ("bearish_cross", "crosses_below"):
                ema_slow = _ema(closes, period2)
                pf = _ema(closes[:-1], period)  if len(closes) > period  else None
                ps = _ema(closes[:-1], period2) if len(closes) > period2 else None
                if all(v is not None for v in [ema_fast, ema_slow, pf, ps]):
                    return pf >= ps and ema_fast < ema_slow
            return False

        if name == "bb":
            bb = _bb(closes)
            if not bb: return False
            curr = closes[-1]
            if sub == "squeeze":             return bb["squeeze"]
            if sub == "price_above_upper":   return curr > bb["upper"]
            if sub == "price_below_lower":   return curr < bb["lower"]
            if sub == "price_near_upper":    return curr > bb["mid"] and curr <= bb["upper"]
            if sub == "price_near_lower":    return curr < bb["mid"] and curr >= bb["lower"]
            return False

        if name == "stochrsi":
            sr = _stochrsi(closes)
            if not sr: return False
            k = sr["k"]
            if sub == "oversold":      return k < 20
            if sub == "overbought":    return k > 80
            if sub == "bullish_cross": return k < 20 and sr.get("prev_k", k) > k
            if sub == "bearish_cross": return k > 80 and sr.get("prev_k", k) < k
            return _cmp(k, op, val)

        if name == "supertrend":
            st = _supertrend(klines)
            if sub == "bearish": return st == "BEARISH"
            return st == "BULLISH"

        if name in ("volume", "volume_spike"):
            return _cmp(_vol_ratio(klines), op, val)

        if name in ("sma", "sma_cross", "sma_ribbon"):
            period  = int(cond.get("period", 200))
            period2 = int(cond.get("period2", 0))
            source  = cond.get("source", "close").lower()
            sub_c   = cond.get("condition", sub or "price_above")

            def _src_be(kl, s):
                if s == "high": return [float(k[2]) for k in kl]
                if s == "low":  return [float(k[3]) for k in kl]
                return [float(k[4]) for k in kl]

            src_data = _src_be(klines, source)
            sma_val  = _sma(src_data, period)
            curr     = float(klines[-1][4])

            if sma_val is None:
                return False

            if sub_c in ("above", "price_above"):
                return curr > sma_val
            if sub_c in ("below", "price_below"):
                return curr < sma_val

            if sub_c in ("above_ribbon", "above_high"):
                sma_high = _sma(_src_be(klines, "high"), period)
                return sma_high is not None and curr > sma_high
            if sub_c in ("below_ribbon", "below_low"):
                sma_low = _sma(_src_be(klines, "low"), period)
                return sma_low is not None and curr < sma_low
            if sub_c == "inside_ribbon":
                sma_high = _sma(_src_be(klines, "high"), period)
                sma_low  = _sma(_src_be(klines, "low"),  period)
                return (sma_high is not None and sma_low is not None
                        and sma_low <= curr <= sma_high)

            if sub_c in ("bullish_cross", "crosses_above") and period2:
                sma_slow = _sma(src_data, period2)
                prev_f   = _sma(src_data[:-1], period)  if len(src_data) > period  else None
                prev_s   = _sma(src_data[:-1], period2) if len(src_data) > period2 else None
                if all(v is not None for v in [sma_val, sma_slow, prev_f, prev_s]):
                    return prev_f <= prev_s and sma_val > sma_slow
            if sub_c in ("bearish_cross", "crosses_below") and period2:
                sma_slow = _sma(src_data, period2)
                prev_f   = _sma(src_data[:-1], period)  if len(src_data) > period  else None
                prev_s   = _sma(src_data[:-1], period2) if len(src_data) > period2 else None
                if all(v is not None for v in [sma_val, sma_slow, prev_f, prev_s]):
                    return prev_f >= prev_s and sma_val < sma_slow

            return _cmp(sma_val, op, val)

        return False  # unsupported indicator sub-type — no fake signals

    # ── Candlestick ─────────────────────────────────────────────────────────────
    if ctype == "candlestick":
        if len(klines) < 2: return False
        # UI sends `pattern`; normalise to `condition`
        sub = cond.get("pattern") or cond.get("condition", "")
        o  = float(klines[-1][1]); h  = float(klines[-1][2])
        l  = float(klines[-1][3]); c  = float(klines[-1][4])
        po = float(klines[-2][1]); pc = float(klines[-2][4])
        body = abs(c - o)
        rng  = h - l
        if not rng: return False
        if sub == "bullish_engulfing": return c > o and pc < po and c > po and o < pc
        if sub == "bearish_engulfing": return c < o and pc > po and c < po and o > pc
        if sub in ("hammer", "pin_bar"):
            lw = min(o, c) - l
            uw = h - max(o, c)
            return lw > body * 2 and uw < body * 0.5
        if sub == "shooting_star":
            uw = h - max(o, c)
            lw = min(o, c) - l
            return uw > body * 2 and lw < body * 0.5
        if sub == "doji": return body / rng < 0.1
        return False

    # ── Consecutive Candles ─────────────────────────────────────────────────────
    if ctype == "consecutive_candles":
        # UI sends `cc_count` and `cc_dir` (bullish/bearish); normalise
        n   = int(cond.get("cc_count") or cond.get("count", 3))
        raw = (cond.get("cc_dir") or cond.get("direction", "green")).lower()
        req = "green" if raw in ("bullish", "green", "up") else "red"
        if len(klines) < n: return False
        for k in klines[-n:]:
            if req == "green" and float(k[4]) <= float(k[1]): return False
            if req == "red"   and float(k[4]) >= float(k[1]): return False
        return True

    # ── VWAP Deviation ──────────────────────────────────────────────────────────
    if ctype == "vwap_deviation":
        vwap = _vwap(klines)
        if vwap is None: return False
        close    = float(klines[-1][4])
        dev_pct  = float(cond.get("vwap_pct", 3))
        side     = cond.get("vwap_side", "below")
        deviation = (close - vwap) / vwap * 100
        if side == "below":  return deviation <= -dev_pct
        if side == "above":  return deviation >= dev_pct
        return abs(deviation) >= dev_pct

    # ── ATR Volatility ──────────────────────────────────────────────────────────
    if ctype == "atr_volatility":
        condition  = cond.get("condition", "contracting")
        multiplier = float(cond.get("multiplier", 1.2))
        period     = 14
        vals = _atr_values(klines, period)
        if len(vals) < period: return False
        curr_atr = vals[-1]
        # Compare current ATR to average of the prior `period` ATR values
        prior_avg = sum(vals[-(period + 1):-1]) / period if len(vals) > period else vals[-1]
        if condition == "expanding":   return curr_atr > prior_avg * multiplier
        if condition == "contracting": return curr_atr < prior_avg / multiplier
        return False

    # ── Keltner Channel ─────────────────────────────────────────────────────────
    if ctype == "keltner":
        kc    = _keltner(klines)
        if kc is None: return False
        close = float(klines[-1][4])
        sub   = cond.get("condition", "squeeze")
        if sub == "above_upper":  return close > kc["upper"]
        if sub == "below_lower":  return close < kc["lower"]
        if sub == "inside_bands": return kc["lower"] <= close <= kc["upper"]
        if sub == "squeeze":
            bb = _bb(_closes(klines))
            if bb is None: return False
            return bb["upper"] < kc["upper"] and bb["lower"] > kc["lower"]
        return False

    # ── Williams %R ─────────────────────────────────────────────────────────────
    if ctype == "williams_r":
        wr  = _williams_r(klines)
        if wr is None: return False
        sub = cond.get("condition", "oversold")
        if sub == "oversold":   return wr < -80
        if sub == "overbought": return wr > -20
        return False

    # ── ADX Filter ──────────────────────────────────────────────────────────────
    if ctype == "adx_filter":
        adx = _adx(klines)
        if adx is None: return False
        sub = cond.get("condition", "trending")
        if sub == "trending": return adx > 25
        if sub == "ranging":  return adx < 25
        return False

    # ── Range Breakout ──────────────────────────────────────────────────────────
    if ctype == "breakout":
        lookback = int(cond.get("bo_lookback", 20))
        bo_pct   = float(cond.get("bo_pct", 1.0))
        bo_dir   = cond.get("bo_dir", "up")
        pivot    = _pivot_range(klines, lookback)
        if pivot is None: return False
        close = float(klines[-1][4])
        if bo_dir in ("up", "either"):
            if close > pivot["high"] * (1 + bo_pct / 100): return True
        if bo_dir in ("down", "either"):
            if close < pivot["low"] * (1 - bo_pct / 100): return True
        return False

    # ── Support / Resistance ────────────────────────────────────────────────────
    if ctype == "support_resistance":
        pivot    = _pivot_range(klines, 20)
        if pivot is None: return False
        close    = float(klines[-1][4])
        sub      = cond.get("condition", "at_support")
        tol      = 0.01  # within 1% of the level counts as "at"
        if sub == "at_support":      return abs(close - pivot["low"])  / pivot["low"]  <= tol
        if sub == "at_resistance":   return abs(close - pivot["high"]) / pivot["high"] <= tol
        if sub == "breakout_above":  return close > pivot["high"] * (1 + tol)
        if sub == "breakout_below":  return close < pivot["low"]  * (1 - tol)
        return False

    # ── Divergence ───────────────────────────────────────────────────────────────
    if ctype == "divergence":
        ind = (cond.get("indicator") or "rsi").lower()
        direction = cond.get("direction", "bullish")
        return _detect_divergence(klines, indicator=ind, direction=direction)

    # ── Fibonacci ────────────────────────────────────────────────────────────────
    if ctype == "fibonacci":
        return _detect_fibonacci(
            klines,
            level_str = str(cond.get("level", "0.618")),
            fib_type  = cond.get("fib_type", "at_retracement"),
        )

    # ── Fair Value Gap (FVG) ─────────────────────────────────────────────────────
    if ctype == "fvg":
        return _detect_fvg(
            klines,
            fvg_dir     = cond.get("fvg_dir", "bullish"),
            min_gap_pct = float(cond.get("min_gap_pct", 0.3)),
        )

    # ── Order Block ──────────────────────────────────────────────────────────────
    if ctype == "order_block":
        return _detect_order_block(klines, ob_type=cond.get("ob_type", "bullish"))

    # ── Market Structure (BOS / CHoCH) ───────────────────────────────────────────
    if ctype == "market_structure":
        return _detect_market_structure(klines, condition=cond.get("condition", "bos_bullish"))

    # ── ICT / Smart Money — new signal types ────────────────────────────────────

    # Inverted Fair Value Gap
    if ctype == "ifvg":
        return _detect_ifvg(
            klines,
            direction    = cond.get("direction", "bullish"),
            min_gap_pct  = float(cond.get("min_gap_pct", 0.3)),
        )

    # Breaker Block
    if ctype == "breaker_block":
        return _detect_breaker_block(
            klines,
            direction = cond.get("direction", "bullish"),
        )

    # Market Structure Shift (tighter lookback than market_structure)
    if ctype == "mss":
        return _detect_mss(
            klines,
            direction = cond.get("direction", "bullish"),
            lookback  = int(cond.get("lookback", 20)),
        )

    # Change of Character (standalone — first break after series of LH or HL)
    if ctype == "choch":
        return _detect_choch(
            klines,
            direction = cond.get("direction", "bullish"),
        )

    # Liquidity Sweep
    if ctype == "liquidity_sweep":
        return _detect_liquidity_sweep(
            klines,
            direction       = cond.get("direction", "bullish"),
            sweep_lookback  = int(cond.get("sweep_lookback", 10)),
        )

    # Mitigation Block
    if ctype == "mitigation_block":
        return _detect_mitigation_block(
            klines,
            direction       = cond.get("direction", "bullish"),
            min_move_mult   = float(cond.get("min_move_mult", 2.5)),
        )

    # Supply / Demand Zone
    if ctype == "supply_demand_zone":
        return _detect_supply_demand_zone(
            klines,
            direction = cond.get("direction", "bullish"),
        )

    # Premium / Discount
    if ctype == "premium_discount":
        return _detect_premium_discount(
            klines,
            direction      = cond.get("direction", "bullish"),
            range_lookback = int(cond.get("range_lookback", 50)),
        )

    # Equilibrium Entry
    if ctype == "equilibrium_entry":
        return _detect_equilibrium(
            klines,
            direction      = cond.get("direction", "bullish"),
            swing_lookback = int(cond.get("swing_lookback", 20)),
        )

    # Pin Bar (standalone — mirrors candlestick pin_bar but as a direct type)
    if ctype == "pin_bar":
        return _detect_pin_bar(
            klines,
            direction = cond.get("direction", "bullish"),
        )

    # Engulfing (standalone)
    if ctype == "engulfing":
        return _detect_engulfing(
            klines,
            direction = cond.get("direction", "bullish"),
        )

    # Inside Bar
    if ctype == "inside_bar":
        return _detect_inside_bar(klines)

    # HH/HL Bullish Structure
    if ctype == "hh_hl_trend":
        return _detect_hh_hl_trend(klines, num_swings=int(cond.get("num_swings", 5)))

    # LH/LL Bearish Structure
    if ctype == "lh_ll_trend":
        return _detect_lh_ll_trend(klines, num_swings=int(cond.get("num_swings", 5)))

    # Fibonacci Retracement at 0.618 / 0.705
    if ctype == "fib_retracement":
        return _detect_fib_retracement(
            klines,
            direction = cond.get("direction", "bullish"),
            lookback  = int(cond.get("lookback", 50)),
        )

    # VWAP Session bounce / rejection
    if ctype == "vwap_session":
        return _detect_vwap_session(
            klines,
            direction = cond.get("direction", "bullish"),
        )

    # ── Session Level ────────────────────────────────────────────────────────────
    if ctype == "session_level":
        ref       = _session_reference(klines, reference=cond.get("reference", "session_low"))
        if ref is None: return False
        close     = float(klines[-1][4])
        tol       = float(cond.get("threshold_pct", 2)) / 100
        side      = cond.get("side", "near")
        dev       = abs(close - ref) / ref
        if side == "near":  return dev <= tol
        if side == "above": return close > ref * (1 + tol)
        if side == "below": return close < ref * (1 - tol)
        return dev <= tol

    # ── Trend Reversal ───────────────────────────────────────────────────────────
    if ctype == "trend_reversal":
        direction = cond.get("condition", cond.get("direction", "bullish"))
        closes = [float(k[4]) for k in klines]
        opens  = [float(k[1]) for k in klines]
        if len(closes) < 25:
            return False
        rsi_s = _rsi_values(closes)
        if len(rsi_s) < 2:
            return False
        rsi      = rsi_s[-1]
        rsi_prev = rsi_s[-2]
        ema21 = _ema_list(closes, 21)
        if len(ema21) < 6:
            return False
        ema_now  = ema21[-1]
        ema_5ago = ema21[-6]
        cur_close  = closes[-1]
        prev_close = closes[-2]
        cur_open   = opens[-1]
        if direction == "bullish":
            return (ema_5ago > ema_now and cur_close <= ema_now * 1.03
                    and cur_close > cur_open and rsi < 50
                    and rsi > rsi_prev and cur_close > prev_close)
        else:
            return (ema_5ago < ema_now and cur_close >= ema_now * 0.97
                    and cur_close < cur_open and rsi > 50
                    and rsi < rsi_prev and cur_close < prev_close)

    # ── Sustained Trend ───────────────────────────────────────────────────────────
    if ctype == "sustained_trend":
        trend_dir      = cond.get("trend_dir", "pump")
        tf_raw         = str(cond.get("timeframe", "1d"))
        periods        = max(2, int(cond.get("periods", 3)))
        min_total_pct  = float(cond.get("min_total_pct", 10.0))
        min_consistent = float(cond.get("min_consistent", 65.0)) / 100.0
        require_active = str(cond.get("require_active", 1)) not in ("0", "false", "False")

        # Map timeframe to hourly candle count per period
        tf_hours = {"1h": 1, "2h": 2, "4h": 4, "1d": 24, "day": 24, "daily": 24}
        candles_per = tf_hours.get(tf_raw, 24)

        # Bucket klines (1h) into higher-timeframe periods
        total_needed = (periods + 2) * candles_per
        if len(klines) < total_needed:
            return False

        # Build period-OHLC from the most recent windows (exclude current forming period)
        buckets = []
        work = list(klines)
        # Most recent incomplete period = last `candles_per` candles (skip it)
        work = work[:-candles_per] if len(work) > candles_per else work
        # Now extract `periods` complete buckets from the tail
        for _ in range(periods):
            if len(work) < candles_per:
                break
            chunk = work[-candles_per:]
            work  = work[:-candles_per]
            o = float(chunk[0][1])
            c = float(chunk[-1][4])
            buckets.append((o, c))
        buckets.reverse()  # oldest first

        if len(buckets) < periods:
            return False

        # Total move
        window_open  = buckets[0][0]
        window_close = buckets[-1][1]
        if window_open <= 0:
            return False
        total_pct = (window_close - window_open) / window_open * 100

        # Consistency
        in_dir = sum(
            1 for (o, c) in buckets
            if (trend_dir == "pump" and c > o) or (trend_dir == "dump" and c < o)
        )
        consistent = in_dir / periods

        # Direction check
        pass_direction  = (total_pct > 0) if trend_dir == "pump" else (total_pct < 0)
        pass_total      = abs(total_pct) >= min_total_pct
        pass_consistent = consistent >= min_consistent

        # Current candle direction
        if require_active and len(klines) >= 2:
            cur_o = float(klines[-1][1])
            cur_c = float(klines[-1][4])
            pass_active = (cur_c >= cur_o) if trend_dir == "pump" else (cur_c <= cur_o)
        else:
            pass_active = True

        return pass_direction and pass_total and pass_consistent and pass_active

    # ── ICT forex day-trade signals (honest sync ports — see helpers above) ──────
    if ctype == "fx_killzone":
        return _bt_killzone_active(int(klines[-1][0]), cond.get("killzone", "london_kz"))

    if ctype == "fx_displacement":
        return _bt_displacement(
            klines,
            direction      = (cond.get("direction") or "any").lower(),
            min_body_ratio = float(cond.get("min_body_ratio") or 3.0),
        )

    if ctype == "fx_ote":
        return _bt_ote(
            klines,
            direction      = (cond.get("direction") or "bullish").lower(),
            swing_lookback = int(cond.get("swing_lookback") or 20),
            fib_low        = float(cond.get("fib_low") or 61.8),
            fib_high       = float(cond.get("fib_high") or 78.6),
        )

    if ctype == "fx_cisd":
        return _bt_cisd(
            klines,
            direction = (cond.get("direction") or "bullish").lower(),
            max_run   = int(cond.get("max_run") or 10),
        )

    if ctype == "fx_sdp":
        return _bt_sdp(
            klines,
            direction      = (cond.get("direction") or "bullish").lower(),
            swing_lookback = int(cond.get("swing_lookback") or 20),
            sweep_window   = int(cond.get("sweep_window") or 5),
            min_body_ratio = float(cond.get("min_body_ratio") or 2.0),
            max_age        = int(cond.get("max_age") or 20),
        )

    # Genuinely unsupported types (need real-time external data feeds):
    #   open_interest  — requires historical OI snapshots (Coinglass API)
    #   liquidation    — requires historical liquidation feed
    #   funding_rate   — requires historical funding rate snapshots
    # Also still no-op (need MTF / order-flow / session-open refs not modelled here):
    #   fx_judas_swing, fx_silver_bullet, fx_breaker, fx_equal_hl, fx_pd_array, fx_po3
    # Return False so the backtest shows 0 trades rather than fake all-pass signals.
    return False


# ── Candle fetching ──────────────────────────────────────────────────────────────
#
# Priority:
#   1. Gate.io Futures (gate.io/api/v4) — covers all USDT perp pairs including
#      low caps (PIPPIN, FARTCOIN, WIF, BONK…). Returns full 30d in one call.
#   2. Kraken (api.kraken.com) — fallback for coins not on Gate.io. Covers
#      BTC, ETH, SOL and major alts with excellent historical depth.
#
# Both sources return 1h OHLCV with proper startTime pagination.
# No proxy is used — if a coin is on neither exchange, we surface an error.


def _to_gateio_pair(symbol: str) -> str:
    """Convert PIPPINUSDT → PIPPIN_USDT for Gate.io futures."""
    base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("_USDT", "")
    return f"{base}_USDT"


# Kraken uses non-standard names for a handful of coins
_KRAKEN_MAP: Dict[str, str] = {
    "BTC": "XBTUSD", "DOGE": "XDGUSD",
}

def _to_kraken_pair(symbol: str) -> str:
    base = symbol.upper().replace("USDT", "").replace("BUSD", "").replace("USD", "")
    return _KRAKEN_MAP.get(base, f"{base}USD")


# Supported timeframes → (gateio_label, kraken_minutes, seconds_per_candle, minutes_per_candle)
_TF_TABLE = {
    "5m":  ("5m",  5,    300,  5),
    "15m": ("15m", 15,   900,  15),
    "1h":  ("1h",  60,  3600,  60),
    "4h":  ("4h",  240, 14400, 240),
}

def _tf_meta(timeframe: str) -> tuple:
    """Return (gateio_label, kraken_minutes, seconds_per_candle, minutes_per_candle).
    Falls back to 1h for any unknown timeframe."""
    return _TF_TABLE.get((timeframe or "1h").lower(), _TF_TABLE["1h"])


async def _fetch_gateio(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
    timeframe: str = "1h",
) -> List:
    """
    Fetch OHLCV candles from Gate.io Futures for the given USDT symbol.
    Gate.io returns up to 999 candles per request and honours the `from`
    parameter, making pagination straightforward.
    Returns [] if the symbol isn't listed on Gate.io.
    """
    gate_label, _, secs_per_candle, _ = _tf_meta(timeframe)
    pair    = _to_gateio_pair(symbol)
    now_s   = int(datetime.now(timezone.utc).timestamp())
    since_s = now_s - days * 86400
    candles: List = []

    # Cap iterations so 5m × 90d (~25 920 candles) still completes
    for _ in range(30):
        try:
            resp = await http_client.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": pair, "interval": gate_label, "limit": 999, "from": since_s},
                timeout=15,
            )
        except Exception as exc:
            logger.debug(f"[Backtest] Gate.io fetch error {pair}: {exc}")
            break

        if resp.status_code != 200:
            break

        data = resp.json()
        if not isinstance(data, list) or not data:
            break

        for k in data:
            candles.append([
                int(k["t"]) * 1000,   # ts → ms
                float(k["o"]),         # open
                float(k["h"]),         # high
                float(k["l"]),         # low
                float(k["c"]),         # close
                float(k.get("v", 0)), # volume
            ])

        last_ts = int(data[-1]["t"])
        if last_ts >= now_s - secs_per_candle:
            break   # reached current time
        since_s = last_ts + secs_per_candle

    return candles


async def _fetch_kraken(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
    timeframe: str = "1h",
) -> List:
    """
    Fetch OHLCV candles from Kraken for the given symbol.
    Returns [] if the symbol isn't listed on Kraken.
    """
    _, kraken_minutes, secs_per_candle, _ = _tf_meta(timeframe)
    pair  = _to_kraken_pair(symbol)
    now_s = int(datetime.now(timezone.utc).timestamp())
    since = now_s - days * 86400
    candles: List = []

    for _ in range(20):
        try:
            resp = await http_client.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": pair, "interval": kraken_minutes, "since": since},
                timeout=15,
            )
            d = resp.json()
        except Exception as exc:
            logger.debug(f"[Backtest] Kraken fetch error {pair}: {exc}")
            break

        if d.get("error"):
            break

        result     = d.get("result", {})
        candle_key = next((k for k in result if k != "last"), None)
        if not candle_key:
            break

        rows = result[candle_key]
        if not rows:
            break

        for c in rows:
            candles.append([
                int(c[0]) * 1000,
                float(c[1]), float(c[2]), float(c[3]), float(c[4]),
                float(c[6]),  # volume is index 6 in Kraken
            ])

        last_ts = result.get("last", 0)
        if not last_ts or last_ts >= now_s - secs_per_candle:
            break
        since = last_ts

    return candles


async def _fetch_tradfi_historical(
    symbol: str,
    asset_class: str,
    days: int,
    timeframe: str,
) -> tuple:
    """Fetch OHLC for a stock / forex / index ticker via yfinance and shape it
    like the crypto path (list of [ts_ms, o, h, l, c, v])."""
    try:
        from app.services.tradfi_prices import get_klines as _tradfi_klines
    except Exception as e:
        logger.debug(f"tradfi import failed: {e}")
        return [], symbol, False
    # Bound limit so a 1m × 90d ask doesn't blow up — yfinance's own period
    # caps already constrain the upstream, this just keeps the array tidy.
    _, _, _, interval_min = _tf_meta(timeframe)
    approx_per_day = max(1, int(round(1440 / max(1, interval_min))))
    limit = min(approx_per_day * max(1, days) + 20, 5000)
    candles = await _tradfi_klines(symbol, asset_class, timeframe, limit)
    if len(candles) >= 60:
        return candles, f"{symbol.upper()} · yfinance {timeframe}", False
    return [], symbol, False


async def _fetch_historical(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
    timeframe: str = "1h",
    asset_class: str = "crypto",
) -> tuple:
    """
    Fetch OHLCV candles for `symbol` over the past `days` days at `timeframe`.

    Returns (candles, source_label, proxy_used):
      candles      — list of [ts_ms, open, high, low, close, volume]
      source_label — human-readable exchange name shown in results
      proxy_used   — always False (we no longer use a BTC proxy)
    """
    # Stocks / forex / indices route through yfinance — no USDT pair lookup.
    if asset_class and asset_class != "crypto":
        return await _fetch_tradfi_historical(symbol, asset_class, days, timeframe)

    # 1. Try Gate.io first — covers virtually all USDT perp pairs
    candles = await _fetch_gateio(symbol, days, http_client, timeframe)
    if len(candles) >= 60:
        base = symbol.upper().replace("USDT", "")
        return candles, f"{base} · Gate.io {timeframe}", False

    # 2. Fall back to Kraken for coins not listed on Gate.io
    candles = await _fetch_kraken(symbol, days, http_client, timeframe)
    if len(candles) >= 60:
        base = symbol.upper().replace("USDT", "")
        return candles, f"{base} · Kraken {timeframe}", False

    return [], symbol, False


# ── Primary condition builder ───────────────────────────────────────────────────
def _build_primary_cond(primary_type: str, primary_cfg: Dict, direction: str) -> Dict:
    """Convert wizard primaryType + primaryCfg into a backtest condition dict."""
    dir_map = {"LONG": "up", "SHORT": "down"}
    pm_dir  = dir_map.get(direction, "any")

    if primary_type == "price_momentum":
        return {
            "type":           "price_momentum",
            "window_minutes": int(primary_cfg.get("pm_window", 15)),
            "operator":       "gt",
            "value":          float(primary_cfg.get("pm_pct", 5)),
            "direction":      pm_dir,
        }
    if primary_type == "volume_spike":
        return {
            "type":       "volume_spike",
            "multiplier": float(primary_cfg.get("multiplier", 2.0)),
        }
    # Indicators handled inside eval_condition_bt under the "indicator" branch
    _INDICATOR_NAMES = {"rsi", "macd", "ema", "sma", "sma_cross", "sma_ribbon", "bb", "stochrsi", "stoch_rsi", "supertrend", "volume"}
    if primary_type in _INDICATOR_NAMES:
        # Normalise stoch_rsi → stochrsi so the evaluator finds the right branch
        name = "stochrsi" if primary_type == "stoch_rsi" else primary_type
        return {"type": "indicator", "name": name, **primary_cfg}

    # Types evaluated directly (not under "indicator" branch)
    _DIRECT_TYPES = {
        "price_momentum", "volume_spike",
        "vwap_deviation", "atr_volatility", "keltner",
        "williams_r", "adx_filter", "breakout",
        "support_resistance", "candlestick", "consecutive_candles",
        # SMC / price-action — implemented via swing/pivot analysis
        "divergence", "fibonacci", "fvg", "order_block",
        "market_structure", "session_level",
        # ICT / Smart Money — new signal types
        "ifvg", "breaker_block", "mss", "choch",
        "liquidity_sweep", "mitigation_block",
        # Supply & Demand
        "supply_demand_zone", "premium_discount", "equilibrium_entry",
        # Price Action (standalone)
        "pin_bar", "engulfing", "inside_bar",
        # Structure
        "hh_hl_trend", "lh_ll_trend", "fib_retracement", "vwap_session",
    }
    if primary_type in _DIRECT_TYPES:
        return {"type": primary_type, **primary_cfg}

    # Genuinely unsupported (open_interest, liquidation, funding_rate)
    return {"type": primary_type, **primary_cfg}


def _build_confirm_cond(conf: Dict) -> Dict:
    """Normalise a wizard confirmation dict into a backtest condition dict."""
    ctype = conf.get("type", "")
    _INDICATOR_NAMES = {"rsi", "macd", "ema", "sma", "sma_cross", "sma_ribbon", "bb", "stochrsi", "stoch_rsi", "supertrend", "volume"}
    if ctype in _INDICATOR_NAMES:
        name = "stochrsi" if ctype == "stoch_rsi" else ctype
        # Strip "type" from conf before spreading to prevent overwriting "indicator"
        conf_clean = {k: v for k, v in conf.items() if k != "type"}
        return {"type": "indicator", "name": name, **conf_clean}
    return conf


# ── Realism / accuracy constants ────────────────────────────────────────────────
# All values are conservative defaults that reflect typical Gate.io / Bitunix
# perpetual-futures conditions. Each is applied symmetrically to LONG and SHORT.
TAKER_FEE_PCT          = 0.05    # 0.05 % per side (taker market order)
ROUND_TRIP_FEE         = TAKER_FEE_PCT * 2     # 0.10 % on notional per round trip
SLIPPAGE_PCT_PER_SIDE  = 0.02    # 0.02 % per side — typical for liquid majors at low size
ROUND_TRIP_SLIPPAGE    = SLIPPAGE_PCT_PER_SIDE * 2  # 0.04 % on notional per round trip
FUNDING_RATE_8H_PCT    = 0.01    # 0.01 % per 8 h on notional — typical neutral perp
MAINTENANCE_MARGIN_PCT = 0.5     # 0.5 % maintenance margin (Gate.io / Bitunix default)
GAP_SLIPPAGE_PCT       = 0.05    # extra 0.05 % on gap-through-stop fills (worst-case)

def _liq_price(direction: str, entry: float, leverage: int) -> float:
    """
    Approximate liquidation price assuming isolated margin.
    Liq distance from entry = (1 - maint_margin/100) / leverage.
    Example: 10× leverage, 0.5 % maint → liq ≈ 9.5 % away from entry.
    """
    if leverage <= 0:
        return 0.0 if direction == "LONG" else float("inf")
    distance_frac = (1.0 - MAINTENANCE_MARGIN_PCT / 100.0) / leverage
    if direction == "LONG":
        return entry * max(0.0, 1.0 - distance_frac)
    return entry * (1.0 + distance_frac)


def _compute_pnl(direction: str, entry: float, exit_price: float,
                 leverage: int, hold_minutes: float = 0.0,
                 include_fees: bool = True, gap_slippage: bool = False) -> float:
    """
    Returns P&L as % of margin, net of fees, slippage and funding.

    All friction items scale with leverage because they're applied to NOTIONAL,
    not margin. Example: 5× leverage, 0.10 % fees + 0.04 % slippage = 0.70 %
    margin drag per round trip even before any price movement.

    `hold_minutes`  : duration the position was held (for funding).
    `gap_slippage` : True when the bar gapped through TP/SL — adds extra
                     slippage because the actual fill is the gapped open price.
    """
    if direction == "LONG":
        raw = (exit_price - entry) / entry * 100 * leverage
    else:
        raw = (entry - exit_price) / entry * 100 * leverage

    if include_fees:
        raw -= ROUND_TRIP_FEE      * leverage   # exchange fees
        raw -= ROUND_TRIP_SLIPPAGE * leverage   # market-order slippage
        if gap_slippage:
            raw -= GAP_SLIPPAGE_PCT * leverage  # gap-through-stop penalty
        if hold_minutes > 0:
            # Funding paid every 8 h on notional value. Approximate as a
            # continuous accrual: hold_minutes / 480 × rate × leverage.
            funding_periods = hold_minutes / (8.0 * 60.0)
            raw -= FUNDING_RATE_8H_PCT * funding_periods * leverage
    return raw


def _compute_stats(trades: List[Dict], interval_min: int) -> Dict:
    """
    Computes summary stats using MULTIPLICATIVE compounding — equity is treated
    like a real account where each trade's % return is applied to the balance
    after the previous trade. This is more accurate than simple sum-of-%
    because losses asymmetrically reduce the base used for the next trade
    (a 50 % loss requires a 100 % gain to recover, not another 50 %).
    """
    # BREAKEVEN is its own outcome (a stop moved to entry that's hit, net ~0). It
    # counts as a CLOSED trade for equity/pnl/drawdown but is NEITHER a win nor a
    # loss, so win_rate is over decided (win+loss) trades only — matching the
    # live/paper executor's three-way label. For strategies WITHOUT stop
    # management (the default) no breakevens are produced, so every metric below
    # is bit-identical to before.
    closed     = [t for t in trades if t["outcome"] in ("WIN", "LOSS", "BREAKEVEN")]
    wins       = [t for t in closed if t["outcome"] == "WIN"]
    losses     = [t for t in closed if t["outcome"] == "LOSS"]
    breakevens = [t for t in closed if t["outcome"] == "BREAKEVEN"]
    decided    = len(wins) + len(losses)

    if not closed:
        return {
            "total_signals": len(trades), "closed_trades": 0,
            "wins": 0, "losses": 0, "breakevens": 0, "win_rate": 0,
            "total_pnl": 0, "total_pnl_simple": 0, "avg_win": 0, "avg_loss": 0,
            "max_drawdown": 0, "avg_hold_minutes": 0, "profit_factor": 0,
            "liquidations": 0,
        }

    avg_win   = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
    avg_loss  = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0

    # Multiplicative equity — accurately reflects compounded account growth
    equity, peak, max_dd = 100.0, 100.0, 0.0
    for t in closed:
        equity *= (1.0 + t["pnl_pct"] / 100.0)
        equity = max(0.0, equity)  # clamp at zero (account can't go negative)
        if equity > peak: peak = equity
        dd = (peak - equity) / peak * 100 if peak > 0 else 0
        if dd > max_dd: max_dd = dd

    total_pnl_compounded = (equity / 100.0 - 1.0) * 100.0
    total_pnl_simple     = sum(t["pnl_pct"] for t in closed)

    avg_hold = sum(t["hold_candles"] for t in closed) / len(closed) * interval_min

    gross_profit = sum(t["pnl_pct"] for t in wins)
    gross_loss   = abs(sum(t["pnl_pct"] for t in losses))
    pf = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)

    # Track forced liquidations as a risk-quality signal
    liqs = sum(1 for t in closed if t.get("exit_reason") == "LIQUIDATION")

    return {
        "total_signals":     len(trades),
        "closed_trades":     len(closed),
        "wins":              len(wins),
        "losses":            len(losses),
        "breakevens":        len(breakevens),
        "win_rate":          round(len(wins) / decided * 100, 1) if decided else 0,
        "total_pnl":         round(total_pnl_compounded, 2),
        "total_pnl_simple":  round(total_pnl_simple, 2),
        "avg_win":           round(avg_win, 2),
        "avg_loss":          round(avg_loss, 2),
        "max_drawdown":      round(max_dd, 2),
        "avg_hold_minutes":  round(avg_hold),
        "profit_factor":     pf,
        "liquidations":      liqs,
    }


def _build_equity_curve(trades: List[Dict]) -> List[Dict]:
    """Multiplicative compounding — matches `_compute_stats`."""
    equity = 100.0
    points = [{"x": 0, "y": round(equity, 2)}]
    for i, t in enumerate(trades):
        if t["outcome"] == "OPEN":
            continue
        equity *= (1.0 + t["pnl_pct"] / 100.0)
        equity = max(0.0, equity)
        points.append({"x": i + 1, "y": round(equity, 2)})
    return points


# ── Main entry point ────────────────────────────────────────────────────────────
async def run_backtest(
    config: Dict,
    days: int = 30,
    precomputed_candles: List | None = None,
    precomputed_source_label: str | None = None,
) -> Dict:
    """
    Run a full strategy backtest against historical OHLCV data.

    config keys (wizard state):
        direction     LONG | SHORT
        primaryType   price_momentum | volume_spike | rsi | macd | ema | ...
        primaryCfg    dict of config for primary signal
        confirms      list of confirmation condition dicts
        tp1           take-profit % (float)
        sl            stop-loss % (float)
        leverage      leverage multiplier (int)
        timeframe     5m | 15m | 1h | 4h
        singleCoin    symbol to test on (e.g. "BTCUSDT"); defaults to BTCUSDT
        maxHoldHours  optional hard exit time (default 48 h)

    `precomputed_candles` lets callers (e.g. the scanner) fetch each
    (coin, timeframe) once and reuse the candles across many strategies,
    avoiding redundant network I/O.

    Returns dict with keys: symbol, days, interval, total_candles,
                            trades, stats, equity_curve, [error]
    """
    direction    = config.get("direction", "LONG")
    tp_pct       = float(config.get("tp1", 3)) / 100
    sl_pct       = float(config.get("sl", 1.5)) / 100
    leverage     = int(config.get("leverage", 10))
    timeframe    = (config.get("timeframe") or "1h").lower()
    # Forex pip-space mode (P5d): when the wizard emits explicit
    # take_profit_pips / stop_loss_pips (forex assets only), we compute
    # TP/SL as `entry ± pips*pip_size` instead of `entry × (1 ± pct)`.
    # This is the right mental model for forex traders and also makes
    # the backtest math identical to the live executor's pip→pct path.
    tp_pips_cfg  = config.get("take_profit_pips") or config.get("tp_pips")
    sl_pips_cfg  = config.get("stop_loss_pips")   or config.get("sl_pips")
    try:
        tp_pips_val = float(tp_pips_cfg) if tp_pips_cfg not in (None, "", 0) else None
        if tp_pips_val is not None and tp_pips_val <= 0:
            tp_pips_val = None
    except (TypeError, ValueError):
        tp_pips_val = None
    try:
        sl_pips_val = float(sl_pips_cfg) if sl_pips_cfg not in (None, "", 0) else None
        if sl_pips_val is not None and sl_pips_val <= 0:
            sl_pips_val = None
    except (TypeError, ValueError):
        sl_pips_val = None
    # ── Trade management: breakeven + trailing stop ─────────────────────────────
    # Modelled to MIRROR the live executor (strategy_executor) so a scanned or
    # built strategy replays in backtest exactly as it trades:
    #   • breakeven_at_pct  — once price covers this % of the entry→TP distance,
    #                         the stop jumps to entry (matches _compute_be_trigger_price
    #                         legacy forex path).
    #   • trailing_stop +    — ratchet the stop behind the best price reached by
    #     trailing_stop_pct    trailing_stop_pct % of price (matches the paper monitor).
    # Both default OFF, so existing callers see zero behaviour change.
    try:
        be_at_pct = float(config.get("breakeven_at_pct") or config.get("breakeven_pct") or 0)
    except (TypeError, ValueError):
        be_at_pct = 0.0
    if be_at_pct < 0:
        be_at_pct = 0.0
    if be_at_pct > 100:
        be_at_pct = 100.0
    trail_on = bool(config.get("trailing_stop"))
    try:
        trail_pct_cfg = float(config.get("trailing_stop_pct") or 0)
    except (TypeError, ValueError):
        trail_pct_cfg = 0.0
    primary_type = config.get("primaryType", "price_momentum")
    primary_cfg  = config.get("primaryCfg") or {}
    confirms     = config.get("confirms") or []

    # Resolve timeframe to actual minutes per candle (was hardcoded to 60)
    _, _, _, interval_min = _tf_meta(timeframe)
    bt_interval  = timeframe

    try:
        from app.services.asset_classes import normalize_asset_class as _norm_ac
        asset_class = _norm_ac(config.get("asset_class"))
    except Exception:
        asset_class = (config.get("asset_class") or "crypto").lower().strip() or "crypto"
    raw_symbol = (config.get("singleCoin") or "").upper().strip()
    if asset_class == "crypto":
        if not raw_symbol:
            raw_symbol = "BTCUSDT"
        symbol = raw_symbol if raw_symbol.endswith("USDT") else raw_symbol + "USDT"
    else:
        # tradfi: symbol is the ticker exactly as it appears in the catalog
        # (e.g. AAPL, EURUSD, SPX) — never append USDT.
        if not raw_symbol:
            return {"error": f"No symbol selected for {asset_class} backtest."}
        symbol = raw_symbol

    # Pip-mode is only enabled when assetclass==forex AND the wizard
    # provided at least one pip value. Otherwise we silently fall through
    # to the percent-based path so legacy forex strategies (or backtests
    # invoked without pips) keep working.
    pip_mode    = (asset_class == "forex") and (tp_pips_val is not None or sl_pips_val is not None)
    pip_size_v  = 0.0
    if pip_mode:
        from app.services.forex_engine import pip_size as _pip_size_fn
        pip_size_v = _pip_size_fn(symbol)
        # If only one of TP/SL is in pips, derive the other from the pct
        # so the user can mix-and-match (rare but supported by the wizard).
        # Use a "reasonable price" placeholder — overwritten per-entry later
        # via _derive_other_pips() so this is just a guard for stats display.

    if precomputed_candles is not None:
        # NOTE: callers (e.g. the scanner) reuse this list across many
        # backtests, so the engine MUST treat it as read-only — every loop
        # below only reads candle[i] / candle[i][k] and never .append/.pop
        # or mutates a candle row.
        candles      = precomputed_candles
        source_label = precomputed_source_label or f"{symbol} · cached {timeframe}"
        proxy_used   = False
    else:
        try:
            async with httpx.AsyncClient() as client:
                candles, source_label, proxy_used = await _fetch_historical(
                    symbol, days, client, timeframe, asset_class=asset_class,
                )
        except Exception as exc:
            return {"error": f"Failed to fetch candle data: {exc}"}

    if len(candles) < 60:
        base = symbol.replace("USDT", "")
        return {"error": f"No historical data found for {base}. Check the symbol name or try a different coin."}

    logger.info(f"[Backtest] {source_label} {timeframe} {days}d: {len(candles)} candles")

    warmup = 60
    primary_cond = _build_primary_cond(primary_type, primary_cfg, direction)
    # Max hold: 48 h by default, configurable via config["maxHoldHours"].
    # FIX: convert hours → CANDLES using the actual interval. Previously
    # max_hold_c was set to max_hold_h (only correct on 1h candles); on 5m
    # candles a "48 h" hold would have closed after 48 candles = 4 h.
    max_hold_h = int(config.get("maxHoldHours") or 48)
    max_hold_c = max(1, int(round(max_hold_h * 60 / interval_min)))

    trades: List[Dict] = []
    open_trade    = None
    prev_cond_met = False  # used for edge detection on the primary condition

    for i in range(warmup, len(candles)):
        candle     = candles[i]
        curr_ts    = int(candle[0])
        curr_open  = float(candle[1])
        curr_high  = float(candle[2])
        curr_low   = float(candle[3])
        curr_close = float(candle[4])

        # ── Always evaluate primary condition for edge tracking ──────────────────
        # We do this even while a trade is open so that when the trade closes,
        # the edge-detection state is up to date and won't re-fire immediately.
        slice_start   = max(0, i + 1 - 200)
        kslice        = candles[slice_start: i + 1]
        curr_cond_met = eval_condition_bt(primary_cond, kslice, interval_min)

        # ── Check open trade: liquidation / timeout / TP / SL ────────────────────
        if open_trade:
            held         = i - open_trade["entry_idx"]
            held_minutes = held * interval_min
            tp_price     = open_trade["tp_price"]
            sl_price     = open_trade["sl_price"]
            liq_price    = open_trade["liq_price"]
            entry_px     = open_trade["entry_price"]

            # ── 1) Liquidation check (HIGHEST priority — happens BEFORE SL/TP) ──
            # If price reached the liquidation level, the exchange force-closes
            # the position at maintenance margin. Loss is approximately the full
            # margin (capped at -100 %) so we model it as a definitive exit.
            if direction == "LONG":
                liq_hit = curr_low <= liq_price
            else:
                liq_hit = curr_high >= liq_price

            # Only count as liquidation if it would happen BEFORE the SL would
            # trigger — for sane configs SL is much closer than liq, so this
            # mostly catches absurd combinations like SL 30 % at 10× leverage.
            if liq_hit and (
                (direction == "LONG"  and liq_price >= sl_price) or
                (direction == "SHORT" and liq_price <= sl_price)
            ):
                # Liquidation realises the maintenance-margin loss; PnL clamped at -100 %
                pnl = max(-100.0, _compute_pnl(direction, entry_px, liq_price,
                                               leverage, held_minutes,
                                               gap_slippage=True))
                trades.append({
                    "entry_ts":     open_trade["entry_ts"],
                    "exit_ts":      curr_ts,
                    "entry_price":  entry_px,
                    "exit_price":   liq_price,
                    "outcome":      "LOSS",
                    "pnl_pct":      round(pnl, 2),
                    "hold_candles": held,
                    "exit_reason":  "LIQUIDATION",
                })
                open_trade    = None
                prev_cond_met = curr_cond_met
                continue

            # ── 2) Max-hold timeout: force close at current close ───────────────
            if held >= max_hold_c:
                pnl     = _compute_pnl(direction, entry_px, curr_close, leverage, held_minutes)
                outcome = "WIN" if pnl >= 0 else "LOSS"
                trades.append({
                    "entry_ts":     open_trade["entry_ts"],
                    "exit_ts":      curr_ts,
                    "entry_price":  entry_px,
                    "exit_price":   curr_close,
                    "outcome":      outcome,
                    "pnl_pct":      round(pnl, 2),
                    "hold_candles": held,
                    "exit_reason":  "TIMEOUT",
                })
                open_trade    = None
                prev_cond_met = curr_cond_met
                continue

            # ── 3) Gap-on-open detection: if the bar opened past TP/SL,
            #       the actual fill is the gapped open price (not the trigger). ──
            gap_exit = None
            if direction == "LONG":
                if curr_open >= tp_price:
                    gap_exit = ("WIN",  curr_open, "TP_GAP")    # gap up past TP
                elif curr_open <= sl_price:
                    gap_exit = ("LOSS", curr_open, "SL_GAP")    # gap down past SL
            else:  # SHORT
                if curr_open <= tp_price:
                    gap_exit = ("WIN",  curr_open, "TP_GAP")    # gap down past TP
                elif curr_open >= sl_price:
                    gap_exit = ("LOSS", curr_open, "SL_GAP")    # gap up past SL

            if gap_exit:
                outcome, exit_price, exit_reason = gap_exit
                pnl = _compute_pnl(direction, entry_px, exit_price, leverage,
                                   held_minutes, gap_slippage=True)
                # A moved (breakeven/trailing) stop can be gapped through while
                # still in profit, so classify a SL-side gap exit by the realised
                # price move vs entry (fee-agnostic label; pnl keeps fee drag).
                if exit_reason == "SL_GAP":
                    _mv  = (exit_price - entry_px) if direction == "LONG" else (entry_px - exit_price)
                    _eps = abs(entry_px) * 1e-7
                    outcome = "BREAKEVEN" if abs(_mv) <= _eps else ("WIN" if _mv > 0 else "LOSS")
                    if open_trade.get("sl_moved"):
                        exit_reason = "TRAIL_BE_GAP"
                trades.append({
                    "entry_ts":     open_trade["entry_ts"],
                    "exit_ts":      curr_ts,
                    "entry_price":  entry_px,
                    "exit_price":   exit_price,
                    "outcome":      outcome,
                    "pnl_pct":      round(pnl, 2),
                    "hold_candles": held,
                    "exit_reason":  exit_reason,
                })
                open_trade    = None
                prev_cond_met = curr_cond_met
                continue

            # ── 4) Standard TP / SL hit detection on wick ───────────────────────
            if direction == "LONG":
                tp_hit = curr_high >= tp_price
                sl_hit = curr_low  <= sl_price
            else:
                tp_hit = curr_low  <= tp_price
                sl_hit = curr_high >= sl_price

            # Smart same-bar TP+SL resolution using the standard intra-bar
            # path heuristic (Bulkowski / TradingView convention):
            #   • Green bar (close > open) → path is OPEN → LOW → HIGH → CLOSE
            #       (price dipped, bottomed, then rallied to close higher)
            #   • Red   bar (close < open) → path is OPEN → HIGH → LOW → CLOSE
            #       (price popped, peaked, then sold off to close lower)
            #   • Doji (close == open) → fall back to conservative "SL first"
            #
            # Apply the path to TP/SL geometry:
            #   LONG  (TP above, SL below)
            #     green → low first → SL hit first  → LOSS
            #     red   → high first → TP hit first → WIN
            #   SHORT (TP below, SL above)
            #     green → low first → TP hit first  → WIN
            #     red   → high first → SL hit first → LOSS
            outcome = None
            exit_price = None
            if tp_hit and sl_hit:
                green = curr_close > curr_open
                red   = curr_close < curr_open
                if direction == "LONG":
                    if red:
                        outcome, exit_price = "WIN",  tp_price   # high → low path
                    else:  # green or doji
                        outcome, exit_price = "LOSS", sl_price   # low → high path (or conservative)
                else:  # SHORT
                    if green:
                        outcome, exit_price = "WIN",  tp_price   # low → high path → TP (below) hit first
                    else:  # red or doji
                        outcome, exit_price = "LOSS", sl_price   # high → low path → SL (above) hit first
            elif tp_hit:
                outcome, exit_price = "WIN",  tp_price
            elif sl_hit:
                outcome, exit_price = "LOSS", sl_price

            if outcome:
                pnl = _compute_pnl(direction, entry_px, exit_price, leverage, held_minutes)
                is_tp = (exit_price == tp_price)
                if is_tp:
                    exit_reason = "TP"
                else:
                    # SL-side exit. A breakeven/trailing-moved stop can be hit in
                    # profit, so classify by the realised price move vs entry
                    # (fee-agnostic label; pnl_pct still carries the fee drag).
                    # Three-way to mirror the live/paper executor exactly: a stop
                    # ratcheted BEYOND entry = WIN, sitting AT entry (true scratch)
                    # = BREAKEVEN, below = LOSS. Tolerance is tick-level (1e-7
                    # relative) matching _classify_sl_outcome in the executor.
                    _mv  = (exit_price - entry_px) if direction == "LONG" else (entry_px - exit_price)
                    _eps = abs(entry_px) * 1e-7
                    outcome = "BREAKEVEN" if abs(_mv) <= _eps else ("WIN" if _mv > 0 else "LOSS")
                    exit_reason = "TRAIL_BE" if open_trade.get("sl_moved") else "SL"
                trades.append({
                    "entry_ts":     open_trade["entry_ts"],
                    "exit_ts":      curr_ts,
                    "entry_price":  entry_px,
                    "exit_price":   exit_price,
                    "outcome":      outcome,
                    "pnl_pct":      round(pnl, 2),
                    "hold_candles": held,
                    "exit_reason":  exit_reason,
                })
                open_trade = None

            # ── 5) Move the stop for breakeven / trailing — applies to the NEXT
            #       bar only (no same-bar look-ahead), mirroring the live monitor.
            if open_trade is not None:
                if open_trade["be_trigger"] is not None and not open_trade["be_done"]:
                    fav = curr_high if direction == "LONG" else curr_low
                    reached = (fav >= open_trade["be_trigger"]) if direction == "LONG" \
                        else (fav <= open_trade["be_trigger"])
                    if reached:
                        open_trade["sl_price"] = entry_px
                        open_trade["be_done"]  = True
                        open_trade["sl_moved"] = True
                _tp = open_trade["trail_pct"]
                if _tp:
                    if direction == "LONG":
                        cand_sl = curr_high * (1 - _tp / 100.0)
                        if cand_sl > open_trade["sl_price"]:
                            open_trade["sl_price"] = cand_sl
                            open_trade["sl_moved"] = True
                    else:
                        cand_sl = curr_low * (1 + _tp / 100.0)
                        if cand_sl < open_trade["sl_price"]:
                            open_trade["sl_price"] = cand_sl
                            open_trade["sl_moved"] = True

            prev_cond_met = curr_cond_met
            continue  # still managing trade or just closed — don't open another this candle

        # ── Edge detection: only fire on FALSE → TRUE transition ─────────────────
        # This is what your real bot does — a signal fires when the condition
        # is freshly met, not while it stays persistently met (e.g. RSI stuck below 40).
        signal_fires  = curr_cond_met and not prev_cond_met
        prev_cond_met = curr_cond_met

        if not signal_fires:
            continue

        # ── Check confirmations (must be currently met) ──────────────────────────
        all_confirm = all(
            eval_condition_bt(_build_confirm_cond(c), kslice, interval_min)
            for c in confirms
        )
        if not all_confirm:
            continue

        # Enter at NEXT candle's open — signal confirmed at close of candle i,
        # earliest realistic fill is open of candle i+1.
        if i + 1 >= len(candles):
            continue

        next_c    = candles[i + 1]
        entry     = float(next_c[1])
        entry_ts  = int(next_c[0])
        entry_idx = i + 1

        if pip_mode:
            # entry ± pips × pip_size — the canonical forex formulation.
            # When only one side was supplied in pips, fall back to the
            # pct value on the other side so the strategy still has both legs.
            tp_dist = (tp_pips_val * pip_size_v) if tp_pips_val is not None else (entry * tp_pct)
            sl_dist = (sl_pips_val * pip_size_v) if sl_pips_val is not None else (entry * sl_pct)
            if direction == "LONG":
                tp_price = entry + tp_dist
                sl_price = entry - sl_dist
            else:
                tp_price = entry - tp_dist
                sl_price = entry + sl_dist
        elif direction == "LONG":
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
        else:
            tp_price = entry * (1 - tp_pct)
            sl_price = entry * (1 + sl_pct)

        # Pre-compute liquidation price so the trade-management loop can check
        # it cheaply on every bar (matters for high-leverage configs).
        liq_price = _liq_price(direction, entry, leverage)

        # Breakeven trigger price (profit-side): entry covered be_at_pct% of the
        # entry→TP distance. tp_price is on the profit side for both directions
        # so (tp_price - entry) is +ve for LONG and -ve for SHORT — correct sign.
        be_trigger = (entry + (be_at_pct / 100.0) * (tp_price - entry)) if be_at_pct > 0 else None
        # Trailing distance (% of price). Honour an explicit pct; else mirror the
        # executor default of half the stop distance when trailing is enabled.
        eff_trail_pct = None
        if trail_on:
            if trail_pct_cfg > 0:
                eff_trail_pct = trail_pct_cfg
            elif sl_pct > 0:
                eff_trail_pct = (sl_pct * 100.0) / 2.0

        open_trade = {
            "entry_ts":    entry_ts,
            "entry_idx":   entry_idx,
            "entry_price": entry,
            "tp_price":    tp_price,
            "sl_price":    sl_price,
            "liq_price":   liq_price,
            "be_trigger":  be_trigger,
            "be_done":     False,
            "trail_pct":   eff_trail_pct,
            "sl_moved":    False,
        }

    # Close any still-open trade at end of data
    if open_trade and candles:
        last_c     = candles[-1]
        last_close = float(last_c[4])
        held       = len(candles) - 1 - open_trade["entry_idx"]
        pnl        = _compute_pnl(direction, open_trade["entry_price"], last_close,
                                  leverage, held * interval_min)
        trades.append({
            "entry_ts":     open_trade["entry_ts"],
            "exit_ts":      int(last_c[0]),
            "entry_price":  open_trade["entry_price"],
            "exit_price":   last_close,
            "outcome":      "OPEN",
            "pnl_pct":      round(pnl, 2),
            "hold_candles": held,
            "exit_reason":  "END_OF_DATA",
        })

    # Tag each trade with its pip move so the UI / stats can display
    # pips-per-trade alongside %PnL. Direction-aware: LONG profit = up,
    # SHORT profit = down. Only meaningful in pip_mode (forex).
    if pip_mode and pip_size_v > 0:
        for t in trades:
            try:
                ep = float(t["entry_price"]); xp = float(t["exit_price"])
                raw = (xp - ep) if direction == "LONG" else (ep - xp)
                t["pip_move"] = round(raw / pip_size_v, 1)
            except Exception:
                t["pip_move"] = 0.0

    stats        = _compute_stats(trades, interval_min)
    equity_curve = _build_equity_curve(trades)

    if pip_mode:
        closed_p = [t for t in trades if t.get("outcome") in ("WIN", "LOSS", "BREAKEVEN")]
        wins_p   = [t for t in closed_p if t["outcome"] == "WIN"]
        losses_p = [t for t in closed_p if t["outcome"] == "LOSS"]
        total_pips_v = sum(float(t.get("pip_move") or 0) for t in closed_p)
        stats["pip_mode"]            = True
        stats["pip_size"]            = pip_size_v
        stats["total_pips"]          = round(total_pips_v, 1)
        stats["avg_pips_per_trade"]  = round(total_pips_v / len(closed_p), 1) if closed_p else 0.0
        stats["avg_pips_win"]        = round(sum(float(t.get("pip_move") or 0) for t in wins_p)   / len(wins_p),   1) if wins_p   else 0.0
        stats["avg_pips_loss"]       = round(sum(float(t.get("pip_move") or 0) for t in losses_p) / len(losses_p), 1) if losses_p else 0.0
        stats["tp_pips_configured"]  = tp_pips_val
        stats["sl_pips_configured"]  = sl_pips_val

    # Format trade timestamps for display
    display_trades = []
    for t in trades:
        entry_dt = datetime.fromtimestamp(t["entry_ts"] / 1000, tz=timezone.utc)
        exit_dt  = datetime.fromtimestamp(t["exit_ts"]  / 1000, tz=timezone.utc)
        display_trades.append({
            **t,
            "entry_date": entry_dt.strftime("%b %d %H:%M"),
            "exit_date":  exit_dt.strftime("%b %d %H:%M"),
        })

    return {
        "symbol":        source_label,
        "days":          days,
        "interval":      bt_interval,
        "total_candles": len(candles),
        "trades":        display_trades,
        "stats":         stats,
        "equity_curve":  equity_curve,
        "fees_included": True,
        "fee_pct":       ROUND_TRIP_FEE,
        "max_hold_h":    max_hold_h,
        # Accuracy model — frontend can show this so users know what's modelled
        "accuracy_model": {
            "fees_pct_round_trip":     ROUND_TRIP_FEE,
            "slippage_pct_round_trip": ROUND_TRIP_SLIPPAGE,
            "gap_slippage_pct":        GAP_SLIPPAGE_PCT,
            "funding_rate_8h_pct":     FUNDING_RATE_8H_PCT,
            "maintenance_margin_pct":  MAINTENANCE_MARGIN_PCT,
            "compounding":             "multiplicative",
            "entry_fill":              "next_bar_open + slippage",
            "tp_sl_fill":              "wick_trigger + slippage (gap_open if gapped)",
            "same_bar_tp_sl":          "bar_color heuristic (green=SL_first for LONG)",
            "liquidation_modeled":     True,
            "pip_mode":                pip_mode,
            "pip_size":                pip_size_v if pip_mode else None,
        },
    }
