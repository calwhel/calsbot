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
def eval_condition_bt(cond: Dict, klines: List, interval_min: int = 5) -> bool:
    """
    Evaluate a single condition against a historical candle slice.
    Returns True for unsupported condition types (pass-through) so they
    don't falsely block signals during replay.
    """
    ctype = cond.get("type", "")

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

    # Genuinely unsupported types (need real-time external data feeds):
    #   open_interest  — requires historical OI snapshots (Coinglass API)
    #   liquidation    — requires historical liquidation feed
    #   funding_rate   — requires historical funding rate snapshots
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


async def _fetch_gateio(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
) -> List:
    """
    Fetch 1h candles from Gate.io Futures for the given USDT symbol.
    Gate.io returns up to 999 candles per request and honours the `from`
    parameter, making pagination straightforward.
    Returns [] if the symbol isn't listed on Gate.io.
    """
    pair    = _to_gateio_pair(symbol)
    now_s   = int(datetime.now(timezone.utc).timestamp())
    since_s = now_s - days * 86400
    candles: List = []

    for _ in range(10):
        try:
            resp = await http_client.get(
                "https://api.gateio.ws/api/v4/futures/usdt/candlesticks",
                params={"contract": pair, "interval": "1h", "limit": 999, "from": since_s},
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
        if last_ts >= now_s - 3600:
            break   # reached current time
        since_s = last_ts + 3600

    return candles


async def _fetch_kraken(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
) -> List:
    """
    Fetch 1h candles from Kraken for the given symbol.
    Returns [] if the symbol isn't listed on Kraken.
    """
    pair  = _to_kraken_pair(symbol)
    now_s = int(datetime.now(timezone.utc).timestamp())
    since = now_s - days * 86400
    candles: List = []

    for _ in range(15):
        try:
            resp = await http_client.get(
                "https://api.kraken.com/0/public/OHLC",
                params={"pair": pair, "interval": 60, "since": since},
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
        if not last_ts or last_ts >= now_s - 3600:
            break
        since = last_ts

    return candles


async def _fetch_historical(
    symbol: str,
    days: int,
    http_client: httpx.AsyncClient,
) -> tuple:
    """
    Fetch 1h OHLCV candles for symbol over the past `days` days.

    Returns (candles, source_label, proxy_used):
      candles      — list of [ts_ms, open, high, low, close, volume]
      source_label — human-readable exchange name shown in results
      proxy_used   — always False (we no longer use a BTC proxy)
    """
    # 1. Try Gate.io first — covers virtually all USDT perp pairs
    candles = await _fetch_gateio(symbol, days, http_client)
    if len(candles) >= 60:
        base = symbol.upper().replace("USDT", "")
        return candles, f"{base} · Gate.io", False

    # 2. Fall back to Kraken for coins not listed on Gate.io
    candles = await _fetch_kraken(symbol, days, http_client)
    if len(candles) >= 60:
        base = symbol.upper().replace("USDT", "")
        return candles, f"{base} · Kraken", False

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
    closed = [t for t in trades if t["outcome"] in ("WIN", "LOSS")]
    wins   = [t for t in closed if t["outcome"] == "WIN"]
    losses = [t for t in closed if t["outcome"] == "LOSS"]

    if not closed:
        return {
            "total_signals": len(trades), "closed_trades": 0,
            "wins": 0, "losses": 0, "win_rate": 0,
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
        "win_rate":          round(len(wins) / len(closed) * 100, 1),
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
async def run_backtest(config: Dict, days: int = 30) -> Dict:
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
        timeframe     5m | 15m | 1h | ...
        singleCoin    symbol to test on (e.g. "BTCUSDT"); defaults to BTCUSDT

    Returns dict with keys: symbol, days, interval, total_candles,
                            trades, stats, equity_curve, [error]
    """
    direction    = config.get("direction", "LONG")
    tp_pct       = float(config.get("tp1", 3)) / 100
    sl_pct       = float(config.get("sl", 1.5)) / 100
    leverage     = int(config.get("leverage", 10))
    timeframe    = config.get("timeframe", "5m")
    primary_type = config.get("primaryType", "price_momentum")
    primary_cfg  = config.get("primaryCfg") or {}
    confirms     = config.get("confirms") or []

    # Backtest always uses 1h candles via Kraken (proper historical pagination)
    bt_interval  = "1h"
    interval_min = 60  # minutes per candle

    raw_symbol = (config.get("singleCoin") or "BTCUSDT").upper().strip()
    symbol = raw_symbol if raw_symbol.endswith("USDT") else raw_symbol + "USDT"

    try:
        async with httpx.AsyncClient() as client:
            candles, source_label, proxy_used = await _fetch_historical(symbol, days, client)
    except Exception as exc:
        return {"error": f"Failed to fetch candle data: {exc}"}

    if len(candles) < 60:
        base = symbol.replace("USDT", "")
        return {"error": f"No historical data found for {base}. Check the symbol name or try a different coin."}

    logger.info(f"[Backtest] {source_label} 1h {days}d: {len(candles)} candles")

    warmup = 60
    primary_cond = _build_primary_cond(primary_type, primary_cfg, direction)
    # Max hold: 48 h by default, configurable via config["maxHoldHours"].
    max_hold_h = int(config.get("maxHoldHours") or 48)
    max_hold_c = max(1, max_hold_h)

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
                trades.append({
                    "entry_ts":     open_trade["entry_ts"],
                    "exit_ts":      curr_ts,
                    "entry_price":  entry_px,
                    "exit_price":   exit_price,
                    "outcome":      outcome,
                    "pnl_pct":      round(pnl, 2),
                    "hold_candles": held,
                    "exit_reason":  "TP" if outcome == "WIN" else "SL",
                })
                open_trade = None

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

        if direction == "LONG":
            tp_price = entry * (1 + tp_pct)
            sl_price = entry * (1 - sl_pct)
        else:
            tp_price = entry * (1 - tp_pct)
            sl_price = entry * (1 + sl_pct)

        # Pre-compute liquidation price so the trade-management loop can check
        # it cheaply on every bar (matters for high-leverage configs).
        liq_price = _liq_price(direction, entry, leverage)

        open_trade = {
            "entry_ts":    entry_ts,
            "entry_idx":   entry_idx,
            "entry_price": entry,
            "tp_price":    tp_price,
            "sl_price":    sl_price,
            "liq_price":   liq_price,
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

    stats        = _compute_stats(trades, interval_min)
    equity_curve = _build_equity_curve(trades)

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
        "interval":      "1h",
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
        },
    }
